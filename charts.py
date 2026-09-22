import re
import sqlite3
import subprocess
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pandas.core.frame import DataFrame

DB_PATH = "brew_stats.db"
conn = sqlite3.connect(DB_PATH)

QUERY = """
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


class ChartType(str, Enum):
    """Chart.js types whose DataFrame layout this notebook supports."""

    BAR = "bar"
    LINE = "line"


@dataclass(frozen=True)
class ChartDef:
    id: str
    title: str
    chart_type: ChartType
    query: str
    description: str | None = None
    value_label: str | None = None


def chartjs_record(
    df: DataFrame,
    *,
    title: str,
    chart_type: ChartType,
    value_column: str = "count",
    category_column: str = "name",
    date_column: str = "date",
    value_label: str | None = None,
    description: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one analysis DataFrame into a self-contained Chart.js record.

    A line chart expects long-form date, category, and value columns. It emits one
    dataset per category and uses ISO-8601 date labels. A missing category/date
    combination becomes ``None`` so ``json.dumps`` writes ``null``.

    A bar chart expects one category/value row per bar. The value may be an
    absolute count or a percentage; its meaning is supplied by ``value_label``.
    """
    chart_type = ChartType(chart_type)
    chart_id = _slugify_title(title)

    if chart_type is ChartType.LINE:
        data = _time_series_chart_data(
            df,
            date_column=date_column,
            category_column=category_column,
            value_column=value_column,
        )
    else:
        data = _category_values_chart_data(
            df,
            category_column=category_column,
            value_column=value_column,
            value_label=value_label,
        )

    record: dict[str, Any] = {
        "id": chart_id,
        "title": title,
        "config": {
            "type": chart_type.value,
            "data": data,
            "options": dict(options or {}),
        },
    }
    if description is not None:
        record["description"] = description

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

CHARTS: list[ChartDef] = []
