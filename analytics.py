# %% imports, constants, helper functions

import sqlite3
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import EngFormatter
from pandas.core.frame import DataFrame

DB_PATH = "brew_stats.db"
conn = sqlite3.connect(DB_PATH)
QUERY = """
SELECT counts.date, names.name, counts.count
FROM counts
JOIN names ON names.name_id = counts.name_id
WHERE names.name {}
ORDER BY counts.date
"""

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

def top_trend(df: DataFrame, title: str | None = None, limit: int | None = 5) -> None:
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    latest_date = df["date"].max()
    top = df[df["date"] == latest_date].nlargest(limit, "count")["name"]
    plot_df = df[df["name"].isin(top)]

    ax = sns.lineplot(
        data=plot_df,
        x="date",
        y="count",
        hue="name",
        hue_order=top,
        style="name",
        markers=True,
        dashes=False,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.yaxis.set_major_formatter(EngFormatter())
    plt.xticks(rotation=90)
    plt.ylabel("installs")
    plt.title(f"Top {min(limit, df['name'].nunique())} {title if title else ''}")
    plt.show()

def merge_names(
    df: pd.DataFrame,
    name_col: str = "name",
    count_col: str = "count",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Merge names containing '@' into their base name and aggregate counts
    per name per date.

    Args:
        df: Input DataFrame.
        name_col: Column containing names (e.g. 'node@22').
        count_col: Column with values to sum.
        date_col: Column to group dates by.

    Returns:
        A new DataFrame with merged names and summed counts.
    """
    result = df.copy()
    result[name_col] = result[name_col].str.split("@").str[0]
    return result.groupby([name_col, date_col], as_index=False)[count_col].sum()

# %% top gainers

df = pd.read_sql_query(
    """
    WITH last_two_dates AS (
      SELECT DISTINCT date FROM counts ORDER BY date DESC LIMIT 2
    ),
    prev AS (
      SELECT name_id, count
      FROM counts
      WHERE date = (SELECT MIN(date) FROM last_two_dates)
    ),
    curr AS (
      SELECT name_id, count
      FROM counts
      WHERE date = (SELECT MAX(date) FROM last_two_dates)
    )
    SELECT
      names.name,
      -- prev.count AS prev_count,
      -- curr.count AS curr_count,
      ROUND((curr.count - prev.count) * 100.0 / prev.count, 2) AS pct_growth
    FROM curr
    JOIN prev  ON prev.name_id = curr.name_id
    JOIN names ON names.name_id = curr.name_id
    WHERE prev.count > 750
    ORDER BY pct_growth DESC
    LIMIT 10;
    """,
    conn,
)

sns.barplot(x="pct_growth", y="name", data=df)
plt.ylabel("Package name")
plt.xlabel("Growth percentage (%)")
plt.title("Top gainers last week")
plt.show()

# %% new entries

df = pd.read_sql_query(
    """
    WITH ranked AS (
        SELECT
            counts.name_id,
            names.name,
            counts.date,
            counts.count,
            ROW_NUMBER() OVER (
                PARTITION BY counts.name_id
                ORDER BY counts.date DESC
            ) AS rn,
            MAX(
                CASE
                    WHEN counts.count <> 0 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY counts.name_id
                ORDER BY counts.date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS had_previous_event
        FROM counts
        JOIN names ON names.name_id = counts.name_id
    )
    SELECT
        date,
        name,
        count
    FROM ranked
    WHERE rn = 1
      AND COALESCE(had_previous_event, 0) = 0
    ORDER BY count DESC
    LIMIT 10;
    """,
    conn,
)

sns.barplot(x="count", y="name", data=df)
plt.ylabel("Package name")
plt.xlabel("Installs")
plt.title("Top new entries last week")
plt.show()


# %% terminal emulators over time

match_list = ", ".join(f"'{name}'" for name in brew_search("terminal emulator"))

df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "terminal emulators")

# %% code editors over time

match_list = ", ".join(
    f"'{name}'"
    for name in brew_search("/.*edit.*code.*/") + brew_search("/.*code.*edit.*/")
)

df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "code editors")

# %% coding agents over time

agents = brew_search(
    r"/(?i)^(?!.*menu bar)(?!.*status).*\bcoding\b\s\b(agent|assistant)\b.*/"
)
agents += [
    "antigravity-cli",
    "gemini-cli",
    "charmbracelet/tap/crush",
    "anomalyco/tap/opencode",
]

match_list = ", ".join(f"'{name}'" for name in agents)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "coding agents")
# %% python package managers
pkgs = set(
    brew_search("/(?i)^(?!.*(?:token|dictation)).*LLM.*/")
    + brew_search("/(?i)^.*offline ai.*/")
    + brew_search("/(?i)^(?!.*token).*large language model.*/")
    + ["mlx"]
)

match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)

top_trend(df, "LLM runners")
# %% search tools

pkgs = set(brew_search("/(?i)^(?!.*(?:backend)).*search|find.*/"))
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "search tools")
# %% container runers

pkgs = brew_search(
    "/(?i)(?=.*container)(?=.*(build|run(ner|times?)?|desktop|gui|manag(e|ing)))/"
)

match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)

top_trend(merge_names(df), "container runners")
# %% languages and runtimes

pkgs = set(
    brew_search(
        "/(?i)(?=.*(?:programming|compiler|interpreter|scripting|sdk))(?=.*language)/"
    )
    + brew_search("/(?i)(?=.*javascript)(?=.*runtime)/")
    + ["rust", "typescript"]
)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(merge_names(df), "languages and runtimes", 10)


# %% javascript runtimes

pkgs = brew_search("/(?i)(?=.*javascript)(?=.*runtime)/")
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)

top_trend(merge_names(df), "javascript runtimes")


# %% top python versions

df = pd.read_sql_query(QUERY.format("LIKE 'python@%'"), conn)
top_trend(df, "python versions")
# %% python package managers
pkgs = set(
    brew_search("python package")
    + brew_search("python dependency")
    # + brew_search("python environment")
    + brew_search("conda")
    + ["pixi"]
)
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "python package managers")
# %% compression tools

pkgs = set(
    brew_search("/(?i)^(?!.*(?:image)).*compression.*/") + brew_search("archiver")
)
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "compression packages")


# %% font data over time

df = pd.read_sql_query(QUERY.format("LIKE 'font-%'"), conn)
top_trend(df, "fonts")

# %% browsers over time

match_list = ", ".join(f"'{name}'" for name in brew_search("web browser"))
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "web browsers")


# %% media players

match_list = ", ".join(f"'{name}'" for name in brew_search("media player"))
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "media players")

