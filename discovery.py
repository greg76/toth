# %% imports, constants, helper functions

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from compression import zstd
from IPython.display import display
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
# %% shell descriptions

pkgs = brew_search("/(?i)(?=.*(?:unix|posix|berkeley))(?=.*shell)/")

outliers = """
nushell: Modern shell for the GitHub era
ksh93: KornShell, ksh93
elvish: Friendly and expressive shell
powershell: Command-line shell and scripting language
"""

pkgs += [pkg for line in outliers.splitlines() if (pkg := line.split(":")[0])]

match_list = ", ".join(f"'{name}'" for name in pkgs)
df = pd.read_sql_query(QUERY_TEMPLATE.format(f"names.name IN ({match_list})"), conn)
top_trend(df, "shells")


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
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# %% load and enrich top brew packages with their descriptions

TOP_LIMIT = 500
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 2

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

df["desc"] = df["name"].map(descriptions).fillna("").str.strip()

missing_descriptions = df.loc[df["desc"].eq("")].copy()
print(f"Excluded {len(missing_descriptions)} packages without descriptions:")
display(missing_descriptions)
# Keep row positions aligned with embeddings after excluding missing descriptions.
df = df.loc[df["desc"].ne("")].reset_index(drop=True)

display(df)

# %% sentence transformer based clustering

# 1. Setup & Embeddings

if len(df) < MIN_CLUSTER_SIZE:
    raise ValueError("Too few packages with descriptions to cluster.")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["desc"].tolist(), normalize_embeddings=True)

# 2. HDBSCAN Clustering via Precomputed Cosine Distance

distance_matrix = cosine_distances(embeddings).astype("float64")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="precomputed",
    cluster_selection_method="eom",
)
df["cluster"] = clusterer.fit_predict(distance_matrix)
print(
    f"{df.loc[df['cluster'] != -1, 'cluster'].nunique()} clusters; "
    f"{df['cluster'].eq(-1).sum()} of {len(df)} described packages unclustered."
)

# 3. Function to Extract Top Keywords Per Cluster


# Fit once across all described packages so common corpus terms get less weight.
vec = TfidfVectorizer(stop_words="english")
try:
    tfidf_matrix = vec.fit_transform(df["desc"])
except ValueError as error:
    if "empty vocabulary" not in str(error):
        raise
    tfidf_matrix = None


def get_cluster_keywords(row_indices, top_n=3):
    """Summarize a cluster using TF-IDF weights from the full corpus."""
    if tfidf_matrix is None:
        return "General"

    scores = np.asarray(tfidf_matrix[row_indices].mean(axis=0)).ravel()
    words = vec.get_feature_names_out()
    top_indices = scores.argsort()[::-1]
    top_indices = top_indices[scores[top_indices] > 0][:top_n]
    return ", ".join(words[top_indices]) or "General"


# 4. Generate Summary DataFrame (Ordered by total download counts)

cluster_summary = []

for cluster_id, group in df.groupby("cluster"):
    # Sort individual packages inside the cluster by count descending
    sorted_group = group.sort_values(by="count", ascending=False)

    if cluster_id == -1:
        label = "Noise / Outliers (Unclustered)"
    else:
        keywords = get_cluster_keywords(sorted_group.index.to_numpy(), top_n=3)
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
    # Remove numerical asymmetry before converting to condensed distances.
    centroid_distances = (centroid_distances + centroid_distances.T) / 2
    np.fill_diagonal(centroid_distances, 0)

    # Human-readable labels: reuse the TF-IDF keywords from summary_df
    label_map = dict(zip(summary_df["cluster_id"], summary_df["cluster_description"]))
    labels = []
    for cid in cluster_ids:
        kw = label_map[cid].split("[")[-1].rstrip("]")  # extract keyword part
        labels.append(f"C{cid}: {kw[:40]}")

    n = len(labels)

    # Average linkage orders similar clusters together without changing values.
    order = leaves_list(
        linkage(squareform(centroid_distances), method="average", optimal_ordering=True)
    )
    ordered_labels = [labels[i] for i in order]
    heatmap_size = max(8, 0.35 * n)
    fig, hax = plt.subplots(figsize=(heatmap_size + 3, heatmap_size))
    sns.heatmap(
        centroid_distances[np.ix_(order, order)],
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        cmap="viridis",
        vmin=0,
        vmax=2,
        square=True,
        cbar_kws={"label": "Cosine distance (0 = identical direction; 2 = opposite)"},
        ax=hax,
    )
    hax.set_title("Cluster centroid distances (lower = more similar)")
    hax.tick_params(axis="both", labelsize=9)
    plt.setp(hax.get_xticklabels(), rotation=90)
    plt.setp(hax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    plt.show()

    # 2D "map" of the clusters via MDS on the centroid distance matrix
    mds = MDS(
        n_components=2,
        metric="precomputed",
        init="classical_mds",
        random_state=42,
    )
    coords = mds.fit_transform(centroid_distances)
    pairs = np.triu_indices(n, k=1)
    original_distances = centroid_distances[pairs]
    projected_distances = euclidean_distances(coords)[pairs]
    projected_sum_squares = np.sum(projected_distances**2)
    stress = (
        np.sqrt(
            np.sum((original_distances - projected_distances) ** 2)
            / projected_sum_squares
        )
        if projected_sum_squares > 0
        else float("nan")
    )
    print(f"2D projection Stress-1: {stress:.3f} (lower = less distance distortion).")

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
        figsize=(12, fig_height),
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

    ax.set_title(f"Approximate cluster map (MDS; Stress-1 = {stress:.3f})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
