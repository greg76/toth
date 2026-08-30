# %% imports, constants, helper functions

import json
import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from compression import zstd

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


# %% load data
with zstd.open(CASK_FILE) as f:
    raw = json.load(f)

    cask_requests = {
        e["cask"]: count
        for e in raw["items"]
        if (count := parse_num(e["count"])) >= MIN_REQUESTS
    }

    print(f"{len(cask_requests)} casks loaded from {CASK_FILE}")

# %% top fonts

font_casks = [
    (str(font), count)
    for font, count in cask_requests.items()
    if font.startswith("font-")
]
top_chart(font_casks, "Fonts")

# %% top terminal emulators

terminal_emulators = brew_search("terminal emulator")
terminal_casks = [
    (str(cask), count)
    for cask, count in cask_requests.items()
    if cask in terminal_emulators
]
top_chart(terminal_casks, "Terminal emulators")

# %% top code editors

editors = brew_search("/.*edit.*code.*/") + brew_search("/.*code.*edit.*/")
editor_casks = [
    (str(cask), count) for cask, count in cask_requests.items() if cask in editors
]
top_chart(editor_casks, "Editors")

# %% browsers

browsers = brew_search("web browser")
browser_casks = [
    (str(cask), count) for cask, count in cask_requests.items() if cask in browsers
]
top_chart(browser_casks, "Web browsers")

# %% list of coding agents

agents = brew_search(
    r"/(?i)^(?!.*menu bar)(?!.*status).*\bcoding\b\s\b(agent|assistant)\b.*/"
)
agents += [
    "antigravity-cli",
    "gemini-cli",
    "charmbracelet/tap/crush",
    "anomalyco/tap/opencode",
]
print(agents)

# %% chart of agents

agent_casks: list[tuple[str, int]] = [
    (str(cask), count) for cask, count in cask_requests.items() if cask in agents
]
top_chart(agent_casks, "Coding agents")

# %% media players

players = brew_search(r"media player")
player_casks = [
    (str(cask), count) for cask, count in cask_requests.items() if cask in players
]

top_chart(player_casks, "Media players")
