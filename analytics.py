# %% imports, constants, helper functions

import sqlite3
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.core.frame import DataFrame

DB_PATH = "brew_stats.db"
conn = sqlite3.connect(DB_PATH)

CASK_FILE = "dumps/20260823cask.zst"
MIN_REQUESTS = 250

def parse_num(formatted_string: str) -> int:
    return int(formatted_string.replace(",", ""))


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


def top_chart(
    casks: list[tuple[str, int]],
    title: str | None = None,
    merge: bool = True,
    limit: int | None = 10,
) -> None:
    df = pd.DataFrame(casks, columns=["Cask name", "Request count"])

    # Extract base name before '@', group, and sum
    if merge:
        base_names = df["Cask name"].str.split("@").str[0]
        df = pd.DataFrame(df.groupby(base_names, as_index=False)["Request count"].sum())

    top_df = df.sort_values(by="Request count", ascending=False).head(limit)

    sns.barplot(x="Request count", y="Cask name", data=top_df)
    plt.title(f"Top {limit} {title if title else ''}")
    plt.show()

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
    plt.xticks(rotation=90)
    plt.title(f"Top {limit} {title if title else ''}")
    plt.show()


# %% font data over time

df = pd.read_sql_query(
    """
    SELECT counts.date, names.name, counts.count
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE names.name LIKE 'font-%'
    ORDER BY counts.date
    """,
    conn,
)
top_trend(df, "fonts")

# %% terminal emulators over time

terminal_emulator_list = ", ".join(
    f"'{name}'" for name in brew_search("terminal emulator")
)

df = pd.read_sql_query(
    f"""
    SELECT counts.date, names.name, counts.count
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE names.name IN ({terminal_emulator_list})
    ORDER BY counts.date
    """,
    conn,
)
top_trend(df, "terminal emulators")

# %% code editors over time

editor_list = ", ".join(
    f"'{name}'"
    for name in brew_search("/.*edit.*code.*/") + brew_search("/.*code.*edit.*/")
)

df = pd.read_sql_query(
    f"""
    SELECT counts.date, names.name, counts.count
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE names.name IN ({editor_list})
    ORDER BY counts.date
    """,
    conn,
)
top_trend(df, "code editors")
# %% browsers over time

editor_list = ", ".join(f"'{name}'" for name in brew_search("web browser"))

df = pd.read_sql_query(
    f"""
    SELECT counts.date, names.name, counts.count
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE names.name IN ({editor_list})
    ORDER BY counts.date
    """,
    conn,
)
top_trend(df, "web browsers")
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

df = pd.read_sql_query(
    f"""
    SELECT counts.date, names.name, counts.count
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE names.name IN ({match_list})
    ORDER BY counts.date
    """,
    conn,
)
top_trend(df, "coding agents")


# %% media players

match_list = ", ".join(f"'{name}'" for name in brew_search("media player"))

df = pd.read_sql_query(
    f"""
    SELECT counts.date, names.name, counts.count
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE names.name IN ({match_list})
    ORDER BY counts.date
    """,
    conn,
)
top_trend(df, "media players")
