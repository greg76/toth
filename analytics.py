# %% imports, constants, helper functions

import json
import sqlite3
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from compression import zstd
from matplotlib.ticker import EngFormatter
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


def top_trend(df: DataFrame, title: str | None = None) -> None:
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    latest_date = df["date"].max()

    # order names by their value on the last date (descending)
    last_values = df[df["date"] == latest_date].set_index("name")["count"]
    order = last_values.sort_values(ascending=False).index.tolist()

    # names with no data on the last date go at the end
    order += [n for n in df["name"].unique() if n not in order]

    ax = sns.lineplot(
        data=df,
        x="date",
        y="count",
        hue="name",
        hue_order=order,
        style="name",
        markers=True,
        dashes=False,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.yaxis.set_major_formatter(EngFormatter())
    plt.xticks(rotation=90)
    plt.ylabel("installs")
    plt.title(f"Top {title if title else 'list'}")
    plt.show()

# %% total install counts from dumps

dumps_dir = (
    Path("dumps")
    if Path("dumps").exists()
    else Path(__file__).resolve().parent / "dumps"
)

data_list = []
for path in sorted(dumps_dir.glob("*.zst")):
    with zstd.open(path, "rt", encoding="utf-8") as f:
        try:
            content = json.load(f)
            category = content.get("category")
            end_date = content.get("end_date")
            total_count = content.get("total_count")
            if (
                category is not None
                and end_date is not None
                and total_count is not None
            ):
                data_list.append(
                    {
                        "name": category,
                        "date": end_date,
                        "count": int(total_count),
                    }
                )
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"Error reading {path}: {e}")

df_dumps = pd.DataFrame(data_list)
df_dumps["date"] = pd.to_datetime(df_dumps["date"])
df_dumps = df_dumps.sort_values("date")

top_trend(df_dumps, "Total Installs over Time by Category")
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
    -- WHERE prev.count > 750
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

# %% code editors over time

match_list = ", ".join(
    f"'{name}'"
    for name in brew_search(r"/.*edit.*code.*/") + brew_search(r"/.*code.*edit.*/")
)

df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "code editors")

# %% coding agents over time

agents = set(
    brew_search(
        r"/(?i)\A(?!.*(?:menu bar|status|pet|memory upgrade)).*\bcod(e|ing)\b\s\b(agent|assistant)\b.*/"
    )
    + brew_search(
        r"/(?i)\A(?!.*(?:review|documentation|language|workout|usage tracker|manage)).*\bAI\b.*(?:programming|code)/"
    )
)

"""
with zstd.open("descriptions.json.zst", "rb") as f:
    descriptions = json.loads(f.read().decode("utf-8"))

max_len = max(len(agent) for agent in agents)

for agent in sorted(agents):
    print(f"{agent:<{max_len}}\t{descriptions.get(agent)}")
"""

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
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "coding harnesses")
# %% agent harnesses
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
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)

top_trend(df, "AI agents")


# %% LLM runners
pkgs = set(
    brew_search("/(?i)^(?!.*(?:token|dictation)).*LLM.*/")
    + brew_search("/(?i)^.*offline ai.*/")
    + brew_search("/(?i)^(?!.*token).*large language model.*/")
    + ["mlx"]
)

match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)

top_trend(df, "LLM runners")

# %% container runers

pkgs = brew_search(
    "/(?i)(?=.*container)(?=.*(build|run(ner|times?)?|desktop|gui|manag(e|ing)))/"
)

match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)

top_trend(df, "container runners")
# %% languages and runtimes

pkgs = set(
    brew_search(
        r"/(?i)(?=.*(?:programming|compiler|interpreter|scripting|sdk))(?=.*language)/"
    )
    + brew_search(r"/(?i)(?=.*javascript)(?=.*runtime)/")
    + ["rust", "typescript"]
)
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "languages and runtimes")


# %% javascript runtimes

pkgs = brew_search("/(?i)(?=.*javascript)(?=.*runtime)/")
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)

top_trend(df, "javascript runtimes")

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

# %% terminal emulators over time

match_list = ", ".join(f"'{name}'" for name in brew_search("terminal emulator"))

df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "terminal emulators")

# %% search tools

pkgs = set(brew_search("/(?i)^(?!.*(?:backend)).*search|find.*/"))
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY.format(f"IN ({match_list})"), conn)
top_trend(df, "search tools")

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
