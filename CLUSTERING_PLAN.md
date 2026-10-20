# Plan: meaningful Homebrew package groups

Implementation handoff, 20 October 2026.

## Problem

We want to discover packages with related purposes. Short descriptions, missing metadata, and a distorted 2D map currently make the groups difficult to trust.

The latest local run found 21 clusters among 460 described packages, left 298 unclustered, and excluded 40 packages without descriptions. The MDS map's Stress-1 was 0.315, indicating substantial distance distortion. These results establish a baseline, not evidence that the revised clustering is better.

## Value

Find related packages and understand each group's purpose through representative members, readable labels, and reproducible comparisons.

Working assumption: similarity means **related functionality**. Dependencies and shared implementation languages may indicate an ecosystem without indicating the same purpose.

## Recommendation

Run a small, measured README enrichment experiment before changing the clustering algorithm. Keep description-only embeddings as the baseline. Evaluate nearest neighbors first, then compare grouping methods using the better representation.

The plotting fixes are already in `discovery.py`: missing descriptions are excluded, labels use corpus-wide TF-IDF, and a distance heatmap supplements the equal-scale MDS map. Further map refinement is a lower priority.

### 1. Preserve the baseline and make quality inspectable

- Freeze the selected 500 packages and cache the input text, model identifier, and clustering settings.
- Export representative and weak members per cluster, with descriptions, HDBSCAN membership strengths, and cluster persistence. These scores describe structural strength, not semantic correctness.
- Choose 20 varied packages and manually judge their five nearest neighbors for related functionality. Retain some examples for a final check after tuning.

**Deliverable:** a repeatable baseline report. Judge improvement by relevant neighbors and coherent membership; report coverage separately. Fewer unclustered packages or lower MDS stress alone does not establish improvement.

### 2. Measure metadata coverage and pilot README enrichment

- Extend metadata loading around `charts.py:get_descriptions()` while preserving existing chart behavior. Retain package type, tap, canonical name, aliases, description, homepage, and upstream source URLs.
- Resolve repositories from `homepage`, `urls.head.url`, and `urls.stable.url`; inspect cask download URLs where applicable. Record unresolved and ambiguous matches. Do not infer identity merely by removing tap prefixes.
- Audit all 500 packages, then fetch READMEs for a varied pilot of about 50 confidently resolved packages. Upstream projects are not always hosted on GitHub, and tap repositories contain packaging definitions.
- Use GitHub's README endpoint, cache by repository, handle rate limits, and retain source URL and revision. Extract introductory prose, features, and use cases within the embedding model's token limit. Exclude badges and installation boilerplate; check package-specific documentation in monorepos.

**Deliverable:** coverage counts, cached excerpts, and reviewed package-to-repository matches. Keep description-only fallback for unavailable documentation. Homebrew exposes these URLs in its [JSON API](https://formulae.brew.sh/docs/api/); GitHub provides a [README endpoint](https://docs.github.com/en/rest/repos/contents#get-a-repository-readme).

### 3. Test whether richer text improves similarity

Compare description alone, name plus description, and name plus description plus README excerpt using `all-MiniLM-L6-v2`. Keep the candidate package set fixed. Report results separately for enriched packages and fallbacks.

If enrichment improves the reviewed neighbors, expand retrieval to the remaining resolved packages. Then compare `all-mpnet-base-v2` on the same text. Change one factor at a time; preserve examples of improvements and regressions.

**Deliverable:** an evidence-backed choice of input text and model, or a decision to retain the baseline.

### 4. Compare grouping and improve labels

Compare HDBSCAN with average-linkage hierarchical clustering using cosine distance. Favor HDBSCAN for tight groups with unassigned packages; favor a hierarchy for exploring broad families and narrower subgroups. Inspect members across nearby parameter settings.

Use representative packages and class-based TF-IDF, which identifies terms distinctive to each cluster, to propose labels. Introduce LLM-generated names only after membership is credible.

**Deliverable:** a recommended grouping method with reviewed examples, coverage, and limitations.

## Defer until the first experiment provides evidence

Dependency-based similarity, full-catalog clustering, dimensionality reduction, embedding fine-tuning, and multi-label classification remain possible directions. Each changes the problem or adds another variable. The most useful next implementation slice is steps 1 and 2, followed by the controlled text comparison in step 3.
