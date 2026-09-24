import json
import re
import sqlite3
import subprocess
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
from pandas.core.frame import DataFrame

DB_PATH = "brew_stats.db"
conn = sqlite3.connect(DB_PATH)
DATA_DATE_FORMAT = "%Y%m%d"

QUERY_LATEST_DATE = "SELECT MAX(date) AS latest_date FROM counts"

QUERY_TEMPLATE = """
WITH cleaned AS (
    SELECT
        counts.date,
        counts.count,
        substr(names.name, 1, instr(names.name || '@', '@') - 1) AS no_at
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE names.name {}
),
normalized AS (
    SELECT
        date,
        count,
        replace(no_at, rtrim(no_at, replace(no_at, '/', '')), '') AS name
    FROM cleaned
),
merged AS (
    SELECT date, name, SUM(count) AS count
    FROM normalized
    GROUP BY date, name
),
top_names AS (
    SELECT name
    FROM merged
    WHERE date = (SELECT MAX(date) FROM counts)
    ORDER BY count DESC, name
    LIMIT 5
)
SELECT date, name, count
FROM merged
WHERE name IN (SELECT name FROM top_names)
ORDER BY date, name
"""

QUERY_NEW_ENTRIES = """
WITH ranked AS (
    SELECT
        counts.name_id,
        names.name,
        counts.date,
        counts.count,
        ROW_NUMBER() OVER (PARTITION BY counts.name_id ORDER BY counts.date DESC) AS rn,
        MAX(CASE WHEN counts.count <> 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY counts.name_id ORDER BY counts.date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            AS had_previous_event
    FROM counts
    JOIN names ON names.name_id = counts.name_id
)
SELECT date, name, count FROM ranked WHERE rn = 1 AND COALESCE(had_previous_event, 0) = 0 ORDER BY count DESC LIMIT 10;
"""

QUERY_GAINERS = """
WITH last_two_dates AS (
    SELECT DISTINCT date FROM counts ORDER BY date DESC LIMIT 2),
prev AS (
    SELECT name_id, count FROM counts WHERE date = (SELECT MIN(date) FROM last_two_dates)
),
curr AS (
    SELECT name_id, count FROM counts WHERE date = (SELECT MAX(date) FROM last_two_dates)
)
SELECT
    names.name, ROUND((curr.count - prev.count) * 100.0 / prev.count, 2) AS pct_growth
FROM curr
JOIN prev  ON prev.name_id = curr.name_id
JOIN names ON names.name_id = curr.name_id
ORDER BY pct_growth DESC
LIMIT 10;
"""


class ChartType(str, Enum):
    """Chart.js types whose DataFrame layout this notebook supports."""
    BAR = "bar"
    LINE = "line"


@dataclass(frozen=True)
class ChartData:
    title: str
    chart_type: ChartType
    df: DataFrame
    description: str | None = None
    value_column: str = "count"
    value_label: str = "Installs"
    category_column: str = "name"
    date_column: str = "date"
    options: Mapping[str, Any] = field(default_factory=dict)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _slugify_title(self.title))


def chartjs_record(
    chart: ChartData,
) -> dict[str, Any]:
    """Convert one chart definition into a self-contained Chart.js record.

    A line chart expects long-form date, category, and value columns. It emits one
    dataset per category and uses ISO-8601 date labels. A missing category/date
    combination becomes ``None`` so ``json.dumps`` writes ``null``.

    A bar chart expects one category/value row per bar. The value may be an
    absolute count or a percentage; its meaning is supplied by ``value_label``.
    """
    chart_type = ChartType(chart.chart_type)

    if chart_type is ChartType.LINE:
        data = _time_series_chart_data(
            chart.df,
            date_column=chart.date_column,
            category_column=chart.category_column,
            value_column=chart.value_column,
        )
    else:
        data = _category_values_chart_data(
            chart.df,
            category_column=chart.category_column,
            value_column=chart.value_column,
            value_label=chart.value_label,
        )

    record: dict[str, Any] = {
        "id": chart.id,
        "title": chart.title,
        "config": {
            "type": chart_type.value,
            "data": data,
            "options": dict(chart.options),
        },
    }
    if chart.description is not None:
        record["description"] = chart.description

    return record


def _time_series_chart_data(
    df: DataFrame,
    *,
    date_column: str,
    category_column: str,
    value_column: str,
) -> dict[str, Any]:
    columns = [date_column, category_column, value_column]
    _require_columns(df, columns)

    data = df.loc[:, columns].copy()
    _require_non_null(data, [date_column, category_column, value_column])
    data[date_column] = pd.to_datetime(data[date_column].astype(str), errors="raise")
    _require_numeric(data, value_column)

    duplicate_rows = data.duplicated([date_column, category_column])
    if duplicate_rows.any():
        raise ValueError(
            "Time-series data must have at most one value per date and category."
        )

    labels = sorted(data[date_column].unique())
    label_strings = [label.strftime("%Y-%m-%d") for label in labels]
    category_order = data[category_column].drop_duplicates().tolist()
    latest_values = data.loc[
        data[date_column] == labels[-1], [category_column, value_column]
    ].set_index(category_column)[value_column]
    categories = latest_values.sort_values(
        ascending=False, kind="stable"
    ).index.tolist()
    categories.extend(
        category for category in category_order if category not in categories
    )
    datasets = []

    for category in categories:
        category_data = data.loc[data[category_column] == category].set_index(
            date_column
        )
        datasets.append(
            {
                "label": str(category),
                "data": [
                    _json_value(category_data.at[label, value_column])
                    if label in category_data.index
                    else None
                    for label in labels
                ],
            }
        )

    return {"labels": label_strings, "datasets": datasets}


def _category_values_chart_data(
    df: DataFrame,
    *,
    category_column: str,
    value_column: str,
    value_label: str | None,
) -> dict[str, Any]:
    columns = [category_column, value_column]
    _require_columns(df, columns)

    data = df.loc[:, columns].copy()
    _require_non_null(data, columns)
    _require_numeric(data, value_column)

    if data[category_column].duplicated().any():
        raise ValueError(
            "Category/value data must have at most one value per category."
        )

    return {
        "labels": [str(category) for category in data[category_column]],
        "datasets": [
            {
                "label": value_label or value_column,
                "data": [_json_value(value) for value in data[value_column]],
            }
        ],
    }


def _slugify_title(title: str) -> str:
    if not title:
        raise ValueError("title must not be empty.")

    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    if not slug:
        raise ValueError("title must contain at least one letter or number.")

    return slug


def _require_columns(df: DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")


def _require_non_null(df: DataFrame, columns: list[str]) -> None:
    empty_columns = [column for column in columns if df[column].isna().any()]
    if empty_columns:
        raise ValueError(f"DataFrame has null values in: {', '.join(empty_columns)}")


def _require_numeric(df: DataFrame, column: str) -> None:
    try:
        df[column] = pd.to_numeric(df[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain numeric values.") from error

    if df[column].isin([float("inf"), float("-inf")]).any():
        raise ValueError(f"{column} must not contain infinite values.")


def _json_value(value: Any) -> int | float:
    return value.item() if hasattr(value, "item") else value


def brew_search(desc: str) -> list[str | None]:
    command = ["brew", "search", "--desc", desc.strip()]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # Triggered if the command returns a non-zero exit status (e.g., brew fails)
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        return []
    except FileNotFoundError:
        # Triggered if 'brew' is not installed or not in the system PATH
        print("Error: The 'brew' command was not found.")
        return []

    return [
        line.partition(":")[0] for line in result.stdout.splitlines() if ":" in line
    ]


#
# --- actual list of charts ---
#

def get_chart_data() -> list[ChartData]:
    chart_data: list[ChartData] = []

    # --- Top gainers ---
    chart_data.append(
        ChartData(
            title="Top gainers",
            chart_type=ChartType.BAR,
            df=pd.read_sql_query(QUERY_GAINERS, conn),
            value_column="pct_growth",
            value_label="Growth percentage (%)",
            description="Packages with the highest recent week-by-week growth and had at least 500 installs.",
        )
    )

    # --- New entries ---
    chart_data.append(
        ChartData(
            title="New entries",
            chart_type=ChartType.BAR,
            df=pd.read_sql_query(QUERY_NEW_ENTRIES, conn),
            description="New packages last week by no of installs.",
        )
    )

    # --- Code editors ---
    match_list = ", ".join(
        f"'{name}'" for name in brew_search(r"/(?i)(?=.*code)(?=.*edit)/")
    )
    chart_data.append(
        ChartData(
            title="Code editors",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Packages whose descriptions indicate that they are used for editing code. With coding harnesses becoming more capable, does coding move away from the editor?",
        )
    )

    # --- coding harnesses ---
    agents = set(
        brew_search(
            r"/(?i)\A(?!.*(?:menu bar|status|pet|memory upgrade)).*\bcod(e|ing)\b\s\b(agent|assistant)\b.*/"
        )
        + brew_search(
            r"/(?i)\A(?!.*(?:review|documentation|language|workout|usage tracker|manage)).*\bAI\b.*(?:programming|code)/"
        )
    )
    agents.update(
        [
            "antigravity",  # Agent orchestration platform
            "antigravity-cli",  # Terminal interface for Antigravity agents
            "gemini-cli",  # Interact with Google Gemini AI models from the command-line
            "charmbracelet/tap/crush",
            "anomalyco/tap/opencode",
            "pi-coding-agent",  # AI agent toolkit
        ]
    )
    match_list = ", ".join(f"'{name}'" for name in agents)
    chart_data.append(
        ChartData(
            title="Coding harnesses",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="AI coding tools designed to work on code, rather than just help you write it.",
        )
    )

    # --- AI agent ---
    pkgs = set(
        brew_search(
            r"/(?i)^(?!.*(?:operator|cod(e|ing)|IDE|scanner|orchestrator|command|container|manage(r)?)).*\bai (agent|assistant)\b/"
        )
        + brew_search("agent runtime")
    )
    pkgs -= {"google-gemini"}
    match_list = ", ".join(
        f"'{name}'"
        for name in pkgs
        if not any(keyword in str(name) for keyword in ["coding", "code"])
    )
    chart_data.append(
        ChartData(
            title="AI agents",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="AI assistants that can use tools, manage tasks and work towards a goal.",
        )
    )

    # --- LLM runners ---
    pkgs = set(
        brew_search("/(?i)^(?!.*(?:token|dictation)).*LLM.*/")
        + brew_search("/(?i)^.*offline ai.*/")
        + brew_search("/(?i)^(?!.*token).*large language model.*/")
        + ["mlx"]
    )
    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="LLM runners",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Tools for downloading, running and interacting with large language models locally.",
        )
    )

    # --- container runners ---
    pkgs = brew_search(
        "/(?i)(?=.*container)(?=.*(build|run(ner|times?)?|desktop|gui|manag(e|ing)))/"
    )
    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="Container runners",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Tools for building, running and managing containers — isolated environments for running software and its dependencies.",
        )
    )

    # --- languages and runtimes ---
    pkgs = set(
        brew_search(
            r"/(?i)(?=.*(?:programming|compiler|interpreter|scripting|sdk))(?=.*language)/"
        )
        + brew_search(r"/(?i)(?=.*javascript)(?=.*runtime)/")
        + ["rust", "typescript"]
    )
    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="Languages and runtimes",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Programming languages, compilers, interpreters and runtimes used to write and run software.",
        )
    )

    # --- javascript runtimes ---
    pkgs = brew_search("/(?i)(?=.*javascript)(?=.*runtime)/")
    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="JavaScript runtimes",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Runtimes for executing JavaScript outside the browser.",
        )
    )

    # --- python package managers ---
    pkgs = set(
        brew_search("python package")
        + brew_search("python dependency")
        # + brew_search("python environment")
        + brew_search("conda")
        + ["pixi"]
    )
    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="Python package managers",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Tools for installing and managing Python tools, packages and their dependencies.",
        )
    )

    # -- static code analysers and linters ---
    pkgs = set(
        brew_search(
            r"/(?i)^(?!.*(claude code|colors|prose|visualization|generator|morphological|compile|driver|computation|statically typed))(?=.*(static|analy[sz]e))(?=.*(\bcode\b|python|php|python|java|swift|Elixir|script|logic))/"
        )
        + brew_search(r"/(?i)^(?!.*(prose|certificate|preview)).*\blinter\b/")
    )
    pkgs.difference_update(["roapi", "sigrok-cli", "memoryanalyzer"])

    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="Code anlyzers and linters",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Code, configuration, or infrastructure definition inspection to identify potential bugs, security issues, and rule violations without executing them. Includes linters, static analyzers, and security-focused analyzers.",
        )
    )

    # --- terminal emulators over time ---
    match_list = ", ".join(f"'{name}'" for name in brew_search("terminal emulator"))
    chart_data.append(
        ChartData(
            title="Terminal emulators",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Alternative terminals that offer more than the OS default. Improved UI and features for managing multiple or remote sessions.",
        )
    )

    # --- search tools ---
    pkgs = set(brew_search("/(?i)^(?!.*(?:backend)).*search|find.*/"))
    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="Search tools",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Command-line tools for finding files, searching code and filtering results. Often faster and more convenient than the traditional Unix tools.",
        )
    )

    # --- compression tools ---
    pkgs = set(
        brew_search("/(?i)^(?!.*(?:image)).*compression.*/") + brew_search("archiver")
    )
    match_list = ", ".join(f"'{name}'" for name in pkgs)
    chart_data.append(
        ChartData(
            title="Compression tools",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn),
            description="Tools for compressing, decompressing, archiving, efficiently storing and moving files around.",
        )
    )

    # --- font data over time ---
    chart_data.append(
        ChartData(
            title="Fonts",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(QUERY_TEMPLATE.format("LIKE 'font-%'"), conn),
            description="Developer and terminal fonts, including monospaced fonts with extra glyphs for richer prompts and text interfaces.",
        )
    )

    return chart_data


def main():
    latest_date = conn.execute(QUERY_LATEST_DATE).fetchone()[0]
    if latest_date is None:
        raise RuntimeError("The counts table does not contain any data.")

    generated_at = datetime.strptime(latest_date, DATA_DATE_FORMAT).replace(
        tzinfo=timezone.utc
    )
    records = [chartjs_record(chart) for chart in get_chart_data()]
    payload = {
        "generatedAt": generated_at.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "charts": records,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
