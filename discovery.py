# %% imports, constants, helper functions

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from compression import zstd
from matplotlib.ticker import EngFormatter
from pandas.core.frame import DataFrame

from charts import (
    QUERY_TEMPLATE,
    ChartType,
    brew_search,
    conn,
    get_chart_data,
    get_descriptions,
)

descriptions = get_descriptions()

def top_trend(df: DataFrame, title: str | None = None) -> None:
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    latest_date = df["date"].max()

    # order names by their value on the last date (descending)
    last_values = df[df["date"] == latest_date].set_index("name")["count"]
    order = last_values.sort_values(ascending=False).index.tolist()

    # names with no data on the last date go at the end
    order += [n for n in df["name"].unique() if n not in order]

    plt.figure(figsize=(9, 4), layout="constrained")
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
    plt.ylabel("Installs")
    plt.title(f"{title if title else 'Top list'}")
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


# %% javascript runtimes

pkgs = brew_search("/(?i)(?=.*web browser)/")
match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY_TEMPLATE.format(f"IN ({match_list})"), conn)
top_trend(df, "web browsers")

# %% published charts

for chart in get_chart_data():
    if chart.chart_type == ChartType.BAR:
        plt.figure(figsize=(9, 4), layout="constrained")
        sns.barplot(x=chart.value_column, y="name", data=chart.df)
        plt.ylabel("Package name")
        plt.xlabel(chart.value_label)
        plt.title(chart.title)
        plt.show()
    elif chart.chart_type == ChartType.LINE:
        top_trend(chart.df, title=chart.title)

# %% load libraries for  clustering descriptions

import hdbscan
from IPython.display import display
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# %% load and enrich top brew packages with their descriptions

TOP_LIMIT = 500

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
