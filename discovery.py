# %% imports, constants, helper functions

import functools
import hashlib
import json
import pickle
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from compression import zstd
from matplotlib.ticker import EngFormatter
from pandas.core.frame import DataFrame

from charts import QUERY, brew_search, conn


def disk_cache(ttl=3600 * 24, cache_dir="/tmp/toth-cache"):
    """Decorator that caches function return values on disk using pickle serialization.

    Hashes positional and keyword arguments using SHA-256 to generate cache keys.
    Cached values are saved as pickle files in `cache_dir` and remain valid for `ttl` seconds.

    Args:
        ttl (int): Time-to-live for cached entries in seconds. Defaults to 24 hours.
        cache_dir (str | Path): Directory where cache files are stored. Defaults to ".cache".

    Returns:
        Callable: A decorator function that wraps the target function with caching logic.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = hashlib.sha256(repr((args, kwargs)).encode()).hexdigest()
            path = cache_dir / key

            if path.exists():
                age = time.time() - path.stat().st_mtime
                if age < ttl:
                    return pickle.loads(path.read_bytes())

            result = func(*args, **kwargs)
            path.write_bytes(pickle.dumps(result))
            return result

        return wrapper

    return decorator

@disk_cache()
def get_descriptions() -> dict[str, Any]:
    URLS = {
        "formulae": "https://formulae.brew.sh/api/formula.json",
        "casks": "https://formulae.brew.sh/api/cask.json",
    }

    descriptions = {}

    for package_type, url in URLS.items():
        print(f"Downloading {package_type}...")
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Python Script)"}
        )

        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))

            for item in data:
                raw_name = item.get("token") or item.get("name")
                desc = item.get("desc")

                if isinstance(raw_name, list) and raw_name:
                    name = raw_name[0]
                else:
                    name = raw_name

                if name and desc:
                    descriptions[name] = desc
    return descriptions


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

# %% load libraries for  clustering descriptions

import hdbscan
from IPython.display import display
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# %% load and enrich top brew packages with their descriptions

TOP_LIMIT = 500

descriptions = get_descriptions()

df = pd.read_sql_query(
    f"""
    SELECT names.name, count
    FROM counts
    JOIN names ON names.name_id = counts.name_id
    WHERE counts.date = (SELECT MAX(date) FROM counts)
    ORDER BY counts.count DESC
    LIMIT {TOP_LIMIT};
    """,
    conn,
)

df["desc"] = df["name"].map(descriptions)

display(df)

# %% sentence transformer based clustering

# 1. Setup & Embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")
df["desc"] = df["desc"].fillna("")
embeddings = model.encode(df["desc"].tolist())

# 2. HDBSCAN Clustering via Precomputed Cosine Distance

distance_matrix = cosine_distances(embeddings).astype("float64")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2, metric="precomputed", cluster_selection_method="eom"
)
df["cluster"] = clusterer.fit_predict(distance_matrix)

# 3. Function to Extract Top Keywords Per Cluster


def get_cluster_keywords(descriptions, top_n=3):
    """Extracts top TF-IDF terms to summarize a group of text descriptions."""

    if not descriptions or all(d == "" for d in descriptions):
        return "Uncategorized"

    vec = TfidfVectorizer(stop_words="english", max_features=10)

    try:
        tfidf_matrix = vec.fit_transform(descriptions)
        # Sum TF-IDF scores across all docs in cluster
        scores = tfidf_matrix.sum(axis=0).A1
        words = vec.get_feature_names_out()
        top_indices = scores.argsort()[-top_n:][::-1]
        return ", ".join([words[i] for i in top_indices])
    except ValueError:
        # Fallback if text is too short or contains only stop words
        return "General"


# 4. Generate Summary DataFrame (Ordered by total download counts)

cluster_summary = []

for cluster_id, group in df.groupby("cluster"):
    # Sort individual packages inside the cluster by count descending
    sorted_group = group.sort_values(by="count", ascending=False)

    # Filter out empty descriptions for keyword extraction
    valid_descs = [d for d in sorted_group["desc"].tolist() if d.strip()]

    if cluster_id == -1:
        label = "Noise / Outliers (Unclustered)"
    else:
        keywords = get_cluster_keywords(valid_descs, top_n=3)
        label = f"Cluster {cluster_id}: [{keywords}]"

    cluster_summary.append(
        {
            "cluster_id": cluster_id,
            "cluster_description": label,
            "total_downloads": sorted_group["count"].sum(),
            "item_count": len(sorted_group),
            "names": sorted_group["name"].tolist(),
        }
    )

summary_df = pd.DataFrame(cluster_summary)

# Separate valid clusters from noise (-1) and rank by total downloads
outliers = summary_df[summary_df["cluster_id"] == -1]
valid_clusters = summary_df[summary_df["cluster_id"] != -1].sort_values(
    by="total_downloads", ascending=False
)

summary_df = pd.concat([valid_clusters, outliers], ignore_index=True)

# Display results nicely
display(summary_df)
# %% visualize how close the clusters are to each other

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

# 1. Compute a centroid embedding per cluster (skip noise, id == -1)
centroids = {}
for cluster_id, group in df[df["cluster"] != -1].groupby("cluster"):
    centroids[cluster_id] = embeddings[group.index].mean(axis=0)

cluster_ids = sorted(centroids)

if len(cluster_ids) < 2:
    print("Need at least 2 non-noise clusters to compare.")
else:
    centroid_matrix = np.vstack([centroids[c] for c in cluster_ids])
    centroid_distances = cosine_distances(centroid_matrix)

    # Human-readable labels: reuse the TF-IDF keywords from summary_df
    label_map = dict(zip(summary_df["cluster_id"], summary_df["cluster_description"]))
    labels = []
    for cid in cluster_ids:
        kw = label_map[cid].split("[")[-1].rstrip("]")  # extract keyword part
        labels.append(f"C{cid}: {kw[:40]}")

    n = len(labels)

    # 2D "map" of the clusters via MDS on the centroid distance matrix
    mds = MDS(
        n_components=2,
        metric="precomputed",
        init="classical_mds",
        random_state=42,
    )
    coords = mds.fit_transform(centroid_distances)

    short_labels = [f"C{cid}" for cid in cluster_ids]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(n)]

    # Scale figure height with the legend length so the scatter plot
    # gets the same vertical space as the legend
    fig_height = max(6, 0.25 * n)

    # Two-column layout: scatter on the left, legend in its own (blank)
    # axes on the right, both spanning the full figure height
    fig, (ax, lax) = plt.subplots(
        1,
        2,
        figsize=(9, fig_height),
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    ax.scatter(coords[:, 0], coords[:, 1], s=80, color=colors)
    for i, short in enumerate(short_labels):
        ax.annotate(
            short, coords[i], textcoords="offset points", xytext=(6, 4), fontsize=9
        )

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[i], markersize=8)
        for i in range(n)
    ]
    lax.legend(legend_handles, labels, loc="upper left", fontsize=9, title="Clusters")
    lax.axis("off")

    ax.set_title("Cluster map (MDS of centroid distances; closer = more similar)")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
