import json
import re
import sqlite3
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.core.frame import DataFrame

DB_PATH = Path(__file__).resolve().parent / "brew_stats.db"


@lru_cache(maxsize=128)
def _compile_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def register_sql_functions(connection: sqlite3.Connection) -> None:
    """Enable SQL REGEXP with Python regex semantics, including lookarounds."""
    connection.create_function(
        "regexp", 2,
        lambda pattern, value: value is not None and bool(_compile_regex(pattern).search(value)),
        deterministic=True,
    )


conn = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
register_sql_functions(conn)
DATA_DATE_FORMAT = "%Y%m%d"

QUERY_LATEST_DATE = "SELECT MAX(date) AS latest_date FROM counts"

QUERY_TEMPLATE = """
WITH cleaned AS (
    SELECT counts.date, counts.count, names.name_id, names.kind_id,
        substr(names.name, 1, instr(names.name || '@', '@') - 1) AS name
    FROM counts JOIN names ON names.name_id = counts.name_id
    WHERE {}
),
merged AS (
    SELECT date, kind_id, name, SUM(count) AS count,
           GROUP_CONCAT(DISTINCT name_id) AS package_ids
    FROM cleaned GROUP BY date, kind_id, name
),
latest_counts AS (
    SELECT kind_id, name, count,
           ROW_NUMBER() OVER (
               PARTITION BY kind_id, name ORDER BY date DESC
           ) AS recency
    FROM merged
),
top_names AS (
    SELECT kind_id, name FROM latest_counts
    WHERE recency = 1
    ORDER BY count DESC, kind_id, name LIMIT 10
)
SELECT date,
    CASE WHEN EXISTS (SELECT 1 FROM merged AS other
                      WHERE other.name = merged.name AND other.kind_id <> merged.kind_id)
         THEN name || CASE kind_id WHEN 0 THEN ' (formula)' ELSE ' (cask)' END
         ELSE name END AS name,
    count, package_ids
FROM merged
WHERE (kind_id, name) IN (SELECT kind_id, name FROM top_names)
ORDER BY date, name
"""


QUERY_NEW_ENTRIES = """
WITH ranked AS (
    SELECT
        counts.name_id,
        names.name,
        names.kind_id,
        counts.date,
        counts.count,
        ROW_NUMBER() OVER (PARTITION BY counts.name_id ORDER BY counts.date DESC) AS rn,
        MAX(CASE WHEN counts.count <> 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY counts.name_id ORDER BY counts.date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            AS had_previous_event
    FROM counts
    JOIN names ON names.name_id = counts.name_id
)
SELECT date, name || CASE WHEN EXISTS (
    SELECT 1 FROM names n WHERE n.name = ranked.name AND n.kind_id <> ranked.kind_id
) THEN CASE kind_id WHEN 0 THEN ' (formula)' ELSE ' (cask)' END ELSE '' END AS name,
count, CAST(name_id AS TEXT) AS package_ids FROM ranked WHERE rn = 1 AND COALESCE(had_previous_event, 0) = 0 ORDER BY count DESC LIMIT 10;
"""

QUERY_GAINERS = """
WITH last_two_dates AS (
    SELECT DISTINCT date FROM counts ORDER BY date DESC LIMIT 2),
merged AS (
    SELECT counts.date, names.kind_id,
        substr(names.name, 1, instr(names.name || '@', '@') - 1) AS name,
        SUM(counts.count) AS count,
        GROUP_CONCAT(DISTINCT names.name_id) AS package_ids
    FROM counts JOIN names ON names.name_id = counts.name_id
    WHERE counts.date IN (SELECT date FROM last_two_dates)
    GROUP BY counts.date, names.kind_id, substr(names.name, 1, instr(names.name || '@', '@') - 1)
),
prev AS (
    SELECT * FROM merged WHERE date = (SELECT MIN(date) FROM last_two_dates)
),
curr AS (
    SELECT * FROM merged WHERE date = (SELECT MAX(date) FROM last_two_dates)
)
SELECT
    curr.name || CASE WHEN EXISTS (
        SELECT 1 FROM merged other WHERE other.name = curr.name AND other.kind_id <> curr.kind_id
    ) THEN CASE curr.kind_id WHEN 0 THEN ' (formula)' ELSE ' (cask)' END ELSE '' END AS name,
    curr.package_ids, ROUND((curr.count - prev.count) * 100.0 / prev.count, 2) AS pct_growth
FROM curr
JOIN prev ON prev.kind_id = curr.kind_id AND prev.name = curr.name
WHERE prev.count > 0 AND (SELECT COUNT(*) FROM last_two_dates) = 2
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

    record["packages"] = package_metadata(chart)
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

    if data.empty:
        return {"labels": [], "datasets": []}

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


def get_descriptions() -> dict[str, str]:
    """Compatibility name lookup for notebooks; production queries retain name IDs."""
    return {name: description for name, description in conn.execute(
        "SELECT name, description FROM names WHERE description IS NOT NULL ORDER BY kind_id"
    )}


def package_metadata(chart: ChartData) -> dict[str, list[dict]]:
    if "package_ids" not in chart.df:
        return {}
    rows = conn.execute(
        "SELECT name_id, kind_id, name, display_name, description, homepage, metadata_refreshed_at FROM names"
    )
    by_id = {str(row[0]): dict(zip(
        ("kind", "name", "display_name", "description", "homepage", "metadata_refreshed_at"),
        (("formula", "cask")[row[1]], *row[2:])
    )) for row in rows}
    result = {}
    for label, group in chart.df.groupby(chart.category_column):
        ids = {value for cell in group["package_ids"] for value in str(cell).split(",")}
        result[str(label)] = [by_id[value] for value in sorted(ids, key=int) if value in by_id]
    return result


# Curated package names and known aliases for public cloud infrastructure and
# managed app platforms. A CLI shared across services appears once.
CLOUD_CLI_PACKAGES = {
    "awscli": "AWS",
    "azure-cli": "Microsoft Azure",
    "gcloud-cli": "Google Cloud",
    "flyctl": "Fly.io",
    "doctl": "DigitalOcean",
    "oci-cli": "Oracle Cloud Infrastructure",
    "hcloud": "Hetzner Cloud",
    "aliyun-cli": "Alibaba Cloud",
    "scw": "Scaleway",
    "openstackclient": "OpenStack",
    "ovh/tap/ovhcloud-cli": "OVHcloud",
    "yandex-cloud-cli": "Yandex Cloud",
    "tencentcloud/tccli/tccli": "Tencent Cloud",
    "linode-cli": "Akamai Cloud Computing",
    "stackitcloud/tap/stackit": "STACKIT",
    "ibm-cloud-cli": "IBM Cloud",
    "vultr-cli": "Vultr",
    "ionos-cloud/ionos-cloud/ionosctl": "IONOS Cloud",
    "civo/tools/civo": "Civo",
    "upcloudltd/tap/upcloud-cli": "UpCloud",
    "exoscale/tap/exoscale-cli": "Exoscale",
    "karanocave/koocli/koocli": "Huawei Cloud",
    "cloudflare-wrangler": "Cloudflare Developer Platform",
    "railway": "Railway",
    "vercel": "Vercel",
    "heroku": "Heroku",
    "render": "Render",
    "netlify-cli": "Netlify",
    "aws-elasticbeanstalk": "AWS Elastic Beanstalk",
    "upsun/tap/upsun-cli": "Upsun",
    "scalingo": "Scalingo",
    "koyeb/tap/koyeb": "Koyeb",
}
CLOUD_CLI_ALIASES = {
    "homebrew/cask/azure-cli": "azure-cli",
    "tencentcloud/tccli/tccli-intl-en": "tencentcloud/tccli/tccli",
    "vultr/vultr-cli/vultr-cli": "vultr-cli",
    "exoscale/tap/exo": "exoscale/tap/exoscale-cli",
    "heroku/brew/heroku": "heroku",
    "upsun/tap/platformsh-cli": "upsun/tap/upsun-cli",
}


CATEGORY_FILTERS = {
    "Code editors": r"""
        description REGEXP '(?i)(?=.*code)(?=.*edit)'
        OR description REGEXP '(?i)\btext editor\b'
        OR name IN ('vim', 'neovim', 'macvim', 'neovide',
                    'd12frosted/emacs-plus/emacs-plus-app')
        OR name GLOB 'd12frosted/emacs-plus/emacs-plus@*'
    """,
    "Coding Agents & Harnesses": r"""
        description REGEXP '(?i)\A(?!.*(?:menu bar|status|pet|memory upgrade)).*\bcod(e|ing)\b\s\b(agent|assistant)\b.*'
        OR description REGEXP '(?i)\A(?!.*(?:review|documentation|language|workout|usage tracker|manage)).*\bAI\b.*(?:programming|code)'
        OR name IN ('antigravity', 'antigravity-cli', 'gemini-cli',
                    'charmbracelet/tap/crush', 'anomalyco/tap/opencode', 'pi-coding-agent')
    """,
    "General-purpose Agents & Harnesses": r"""
        (description REGEXP '(?i)^(?!.*(?:operator|cod(e|ing)|IDE|scanner|orchestrator|command|container|manage(r)?)).*\bai (agent|assistant)\b'
         OR description REGEXP '(?i)agent runtime')
        AND name <> 'google-gemini' AND name NOT LIKE '%coding%' AND name NOT LIKE '%code%'
    """,
    "LLM runners": r"""
        description REGEXP '(?i)^(?!.*(?:token|dictation)).*LLM.*'
        OR description REGEXP '(?i)^.*offline ai.*'
        OR description REGEXP '(?i)^(?!.*token).*large language model.*'
        OR name = 'mlx'
    """,
    "Diagrams as code": r"""
        names.kind_id = 0 AND (
            description REGEXP '(?i)\b(?:diagrams?|diagramming|flowcharts?|uml|graph (?:visuali[sz]ation|layout|rendering))\b'
            OR name IN ('mermaid-cli', 'merman-cli')
        )
    """,
    "Local container platforms": r"""
        description REGEXP '(?i)(?=.*container)(?=.*(build|run(ner|times?)?|desktop|gui|manag(e|ing)))'
        OR name = 'orbstack'
    """,
    "Languages and runtimes": r"""
        description REGEXP '(?i)(?=.*(?:programming|compiler|interpreter|scripting|sdk))(?=.*language)'
        OR description REGEXP '(?i)(?=.*javascript)(?=.*runtime)'
        OR name IN ('rust', 'typescript')
    """,
    "JavaScript runtimes": r"description REGEXP '(?i)(?=.*javascript)(?=.*runtime)'",
    "Python package managers": r"""
        description REGEXP '(?i)python package|python dependency|conda'
        OR name = 'pixi'
    """,
    "Programming error checkers": r"""
        (description REGEXP '(?i)\b(lint|linters?|linting|type checker)\b|\bstatic analysis\b.*\b(?:code|scripts?)\b'
         AND description NOT REGEXP '(?i)\b(?:prose|markdown|yaml|toml|openapi|swagger|dockerfile|kubernetes|terraform|style|conventions|interface builder|sql)\b'
         AND name NOT IN ('ast-grep', 'ktlint'))
        OR name IN ('biome', 'eslint', 'flake8')
    """,
    "Terminal emulators": r"""
        (description REGEXP '(?i)terminal emulator' OR name IN ('warp', 'cmux'))
        AND name <> 'vte3'
    """,
    "Command-line shells": r"""
        description REGEXP '(?i)(?=.*(?:unix|posix|berkeley))(?=.*shell)'
        OR name IN ('nushell', 'ksh93', 'elvish', 'powershell')
    """,
    "Search tools": r"description REGEXP '(?i)^(?!.*(?:backend)).*search|find.*'",
    "Compression tools": r"""
        description REGEXP '(?i)^(?!.*(?:image)).*compression.*'
        OR description REGEXP '(?i)archiver'
    """,
    "Fonts": "name LIKE 'font-%'",
}


def category_query(title: str) -> str:
    # Match metadata once per package, before joining to historical counts.
    return QUERY_TEMPLATE.format(
        f"names.name_id IN (SELECT name_id FROM names WHERE {CATEGORY_FILTERS[title]})"
    )


def _cloud_platform_chart_data() -> DataFrame:
    packages = {name: name for name in CLOUD_CLI_PACKAGES} | CLOUD_CLI_ALIASES
    values = ", ".join("(?, ?)" for _ in packages)
    return pd.read_sql_query(
        f"""
        WITH packages(package_name, cli_name) AS (VALUES {values}),
        merged AS (
            SELECT counts.date, packages.cli_name AS name, SUM(counts.count) AS count,
                   GROUP_CONCAT(DISTINCT names.name_id) AS package_ids
            FROM packages
            JOIN names ON names.name = packages.package_name
            JOIN counts ON counts.name_id = names.name_id
            GROUP BY counts.date, packages.cli_name
        ),
        top_names AS (
            SELECT name FROM merged
            WHERE date = ({QUERY_LATEST_DATE})
            ORDER BY count DESC, name
            LIMIT 10
        )
        SELECT date, name, count, package_ids
        FROM merged
        WHERE name IN (SELECT name FROM top_names)
        ORDER BY date, name
        """,
        conn,
        params=[value for pair in packages.items() for value in pair],
    )


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
            description="Packages with the highest growth in rolling 30-day installs between the two latest snapshots. Versions and release channels are combined.",
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
    chart_data.append(
        ChartData(
            title="Code editors",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Code editors"), conn),
            description="Code and text editors, including selected Vim and Emacs variants. With coding harnesses becoming more capable, does coding move away from the editor?",
        )
    )

    # --- coding harnesses ---
    chart_data.append(
        ChartData(
            title="Coding Agents & Harnesses",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Coding Agents & Harnesses"), conn),
            description="AI coding tools designed to work on code, rather than just help you write it.",
        )
    )

    # --- General-purpose Agents & Harnesses ---
    chart_data.append(
        ChartData(
            title="General-purpose Agents & Harnesses",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("General-purpose Agents & Harnesses"), conn),
            description="AI assistants that can use tools, manage tasks and work towards a goal.",
        )
    )

    # --- LLM runners ---
    chart_data.append(
        ChartData(
            title="LLM runners",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("LLM runners"), conn),
            description="Tools for downloading, running and interacting with large language models locally.",
        )
    )

    # --- diagrams as code ---
    chart_data.append(
        ChartData(
            title="Diagrams as code",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Diagrams as code"), conn),
            description="Command-line tools that turn text into flowcharts, sequence diagrams and network graphs.",
        )
    )

    # --- cloud platform CLIs ---
    chart_data.append(
        ChartData(
            title="Cloud platform CLIs",
            chart_type=ChartType.LINE,
            df=_cloud_platform_chart_data(),
            description=(
                "The 10 most-installed Homebrew CLIs for managing public cloud infrastructure "
                "(IaaS) or deploying apps to managed platforms (PaaS). "
                "Each CLI's count covers all the services it supports."
            ),
        )
    )

    # --- local container platforms ---
    chart_data.append(
        ChartData(
            title="Local container platforms",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Local container platforms"), conn),
            description="Tools that run containers locally or provide a desktop interface for managing a local container runtime.",
        )
    )

    # --- languages and runtimes ---
    chart_data.append(
        ChartData(
            title="Languages and runtimes",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Languages and runtimes"), conn),
            description="Programming languages, compilers, interpreters and runtimes used to write and run software.",
        )
    )

    # --- javascript runtimes ---
    chart_data.append(
        ChartData(
            title="JavaScript runtimes",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("JavaScript runtimes"), conn),
            description="Runtimes for executing JavaScript outside the browser.",
        )
    )

    # --- python package managers ---
    chart_data.append(
        ChartData(
            title="Python package managers",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Python package managers"), conn),
            description="Tools for installing and managing Python tools, packages and their dependencies.",
        )
    )

    # --- programming error checkers ---
    chart_data.append(
        ChartData(
            title="Programming error checkers",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Programming error checkers"), conn),
            description="Selected linters and type checkers that detect syntax, type and other likely programming errors without running the code.",
        )
    )

    # --- terminal emulators over time ---
    chart_data.append(
        ChartData(
            title="Terminal emulators",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Terminal emulators"), conn),
            description="Applications that provide a terminal for local shells and command-line programs, including terminals with tabs, panes or coding agent support.",
        )
    )

    # --- shells ---
    chart_data.append(
        ChartData(
            title="Command-line shells",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Command-line shells"), conn),
            description="Shells for running commands, scripting tasks and interacting with the operating system.",
        )
    )

    # --- search tools ---
    chart_data.append(
        ChartData(
            title="Search tools",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Search tools"), conn),
            description="Command-line tools for finding files, searching code and filtering results. Often faster and more convenient than the traditional Unix tools.",
        )
    )

    # --- compression tools ---
    chart_data.append(
        ChartData(
            title="Compression tools",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Compression tools"), conn),
            description="Tools for compressing, decompressing, archiving, efficiently storing and moving files around.",
        )
    )

    # --- font data over time ---
    chart_data.append(
        ChartData(
            title="Fonts",
            chart_type=ChartType.LINE,
            df=pd.read_sql_query(category_query("Fonts"), conn),
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
    try:
        main()
    finally:
        conn.close()
