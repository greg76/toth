# Clustering experiment progress

20 October 2026. The baseline, text and model experiments, family-aware evaluation,
selective matching, and grouping-method comparison in
[CLUSTERING_PLAN.md](CLUSTERING_PLAN.md) have run. Refer to
[CLUSTERING_RECOMMENDATION.md](CLUSTERING_RECOMMENDATION.md) for the current direction.
New semantic judgments include
assistant review and remain provisional.

## Results

The frozen description-only baseline reproduces 21 clusters, 298 unclustered
packages, and 40 missing descriptions among the top 500 packages for analytics
date `20261018`. Saved vectors, input checksums, model revision, dependency
versions, and settings make the run reproducible. Cluster reports include
representative and weak members, membership strengths, and persistence.

All 21 persistence scores equal 1.0, so that measure cannot rank these clusters.
Thirteen identical-description groups cover 41 packages, including version
families. Duplicates may affect density estimates; a later comparison should
report results by package family as well as by package.

The metadata audit found:

| Result | Packages |
| --- | ---: |
| Matched official formula or cask metadata | 462 |
| Unresolved metadata | 38 |
| Direct GitHub repository candidates | 285 |
| Shared repositories requiring scope review | 20 |
| Conflicting repository URLs | 1 |
| Matched metadata without a GitHub repository candidate | 156 |

Repository rows, including the 38 unresolved metadata rows, total 500. Matches
preserve formula/cask type and aliases. Unresolved tap-qualified names are not
guessed. Current metadata supplies a description for `openssl@3.5`, which was
missing from the frozen lookup; that correction is separate from enrichment.

The pilot selected 50 packages across different regions of the baseline embedding
space. All 50 READMEs downloaded; 49 yielded nonempty, token-limited excerpts.
AltTab's README is image-only. ORAS's excerpt mostly describes release images,
illustrating why nonempty text is not sufficient evidence of useful enrichment.
Automatic repository candidates have not been approved as correct identities.

Inspection exposed extraction errors involving badges, tables of contents,
multiline HTML, and headings containing words such as “installation” or
“compilation.” Those cases were repaired and covered by regression tests.
Extraction remains heuristic and can retain navigation or other irrelevant prose.

## Review artifacts

Generated data is local and ignored by Git. Run `python clustering_experiment.py
all` from the repository root to regenerate it in a new checkout.

- [Experiment report](clustering_runs/baseline/REPORT.md): baseline, coverage,
  representative packages, and weak members.
- [Neighbor judgments](clustering_runs/baseline/neighbor_judgments.csv): 20 anchors,
  five neighbors each, with 15 development anchors and five holdout anchors.
  Fill `related_functionality` with `yes`, `no`, or `uncertain`, and explain
  borderline cases in `notes`.
- [README review](clustering_runs/baseline/readme_review.csv): package identity,
  repository, source revision, and excerpt. Fill `review`, `usable`, and `notes`
  after checking identity, package-specific scope, and useful functional content.
- [Metadata audit](clustering_runs/baseline/metadata_audit.json): all 500 packages,
  matching status, repository signals, and reasons for excluding candidates.

## Next decision

The user labeled all 75 development pairs: 26 relevant, 42 irrelevant, seven
uncertain. Original labels and notes remain unchanged, including the uncertain
MariaDB/MySQL-client pair. The five previously agreed example pairs guide assistant
review but are not counted as extra evaluation observations.

The assistant reviewed all 50 README candidates and retained verbatim functional
passages for 46. Four use name-plus-description fallbacks: ORAS describes release
tags, AltTab is image-only, Railway is dominated by commands, and Azure CLI adds
little functional information. The comparison uses the same 460 described
packages, pinned model, and 15 development anchors. Nine anchors receive excerpts.
Thirty-five new neighbor pairs were assistant-reviewed; seven original uncertain
labels were preserved. User and assistant provenance is recorded separately.

| Representation | Relevant | Irrelevant | Uncertain | Relevant among decisive pairs |
| --- | ---: | ---: | ---: | ---: |
| Description | 26 | 42 | 7 | 38.2% |
| Name plus description | 28 | 38 | 9 | 42.4% |
| Name plus description plus reviewed README | 29 | 39 | 7 | 42.6% |

The denominator excludes uncertain pairs; conservative top-five relevance bounds
are 34.7–44.0%, 37.3–49.3%, and 38.7–48.0%, respectively. These intervals overlap,
and new pairs mainly rely on assistant judgments. This does not establish an
independently validated improvement. Detailed metrics include separate enriched
and fallback anchors, plus user-only label coverage.

Names help Windows App retrieve RustDesk and move MySQL ahead of OpenTofu for
MariaDB. README text adds a related shell-assistance neighbor for
zsh-autosuggestions, but pgvector still retrieves unrelated file-search and
benchmarking tools. No variant reliably solves the sparse or ambiguous cases.

Use name-plus-description as the next experimental baseline. Do not expand README
retrieval yet: its marginal gain does not justify the extra extraction and review
work on this evidence. The stronger model experiment below keeps this text
representation fixed to isolate the effect of changing the embedding model.
No representation has been promoted to `discovery.py` or published charts.

Fifteen original holdout rows have also been labeled and discussed. None are used
by the comparison; select fresh anchors for independent final validation after
choosing the representation. Original holdout entries remain untouched.

Refer to [COMPARISON.md](clustering_runs/baseline/COMPARISON.md) for per-anchor
neighbors and [comparison_judgments.csv](clustering_runs/baseline/comparison_judgments.csv)
for the preserved user judgments and provisional assistant additions.

## Verification

Twenty-four unit tests pass, covering identity separation, aliases, conflicting URLs,
packaging-repository exclusions, README extraction, caching, review preservation,
source-backed excerpts, and explicit treatment of uncertainty and missing labels.
The full pipeline reran from cache without network access. Artifact checks
confirmed the frozen input checksum, vector alignment, unique package identities,
holdout split, and synchronized excerpts in the review export.
Comparison checks confirm the same candidate set, unchanged user inputs, exact
name-plus-description fallback vectors, and cached reruns that preserve reviews.

## Stronger model experiment

Tested `sentence-transformers/all-mpnet-base-v2` at revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130` against the saved MiniLM vectors. The
460 name-plus-description inputs and 15 development anchors are identical.
Twenty-one new neighbor pairs were assistant-reviewed, preserving existing user
labels and uncertainty. No holdout labels were used.

| Model | Relevant | Irrelevant | Uncertain | Relevant among decisive pairs |
| --- | ---: | ---: | ---: | ---: |
| MiniLM | 28 | 38 | 9 | 42.4% |
| MPNet | 29 | 34 | 12 | 46.0% |

MPNet yields one extra relevant top-five result across 75 slots; its higher
decisive rate also reflects more uncertain pairs. Conservative relevance bounds
are 37.3–49.3% for MiniLM and 38.7–54.7% for MPNet. These are uncertainty bounds,
not statistical confidence intervals. New judgments are provisional assistant
assessments, so the gain has not been independently validated.

The clearest gain is for `fizz`: TLS libraries replace `whisper.cpp` and `llvm`,
including NSS, whose TLS support was checked against
[Mozilla's documentation](https://firefox-source-docs.mozilla.org/security/nss/index.html).
For `pgvector`, PostgreSQL packages replace file-search tools, although the user
has left database-server versus extension relationships uncertain. Several
PostgreSQL versions occupy neighboring slots, which limits this apparent gain.
ORAS still incorrectly favors Oracle's `oci-cli`; Windows App still has weak
neighbors. A larger model does not resolve every ambiguous short description.

At unchanged HDBSCAN settings (`min_cluster_size=5`, `min_samples=2`), MiniLM
produces 16 clusters and MPNet 21. Both leave 317 of 460 packages unclustered.
These counts use name-plus-description and are distinct from the original
description-only baseline. MPNet separates TLS libraries from VPN applications,
consistent with the user's preferred boundary. It still groups Docker with
virtualenv and media players with development libraries, so membership remains
imperfect.

MPNet is a reasonable candidate for the next clustering experiment, with modest
evidence of better separation. Keep MiniLM as the control. Before promoting
either model, compare family-aware cluster membership and a fresh set of anchors.
Do not expand README retrieval on the strength of this model-only test.

Refer to [MODEL_COMPARISON.md](clustering_runs/baseline/MODEL_COMPARISON.md) for
neighbors, uncertainty, and cluster counts. Cached model reruns, user-label
preservation, candidate alignment, and normalized MPNet vectors were checked.
The notebook and published charts still use their existing settings.

## Family-aware evaluation and grouping

Qwen3-Embedding-0.6B at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
was tested with one fixed functional-grouping instruction on every package.
Its symmetric vectors support both neighbor comparison and clustering. The
460 described packages become 437 canonical families after removing `@version`
suffixes. Formula/cask type and unresolved tap prefixes remain part of identity.
Candidate family ranking uses the strongest member match, omitting the anchor's
own family. Clustering uses one normalized mean vector per family.

Forty-eight newly exposed family pairs were assistant-reviewed. There are 145
distinct reviewed development pairs in the union: 56 inherit user labels and
89 carry assistant labels. Uncertain and conflicting family labels stay uncertain.
The five explicitly agreed reference pairs are recorded in
`clustering_reference_pairs.json` and checked separately.

Distinct-family top-five results contain 25 relevant MiniLM neighbors, 27 MPNet
neighbors, and 28 Qwen neighbors across 75 slots each. In leave-one-anchor-out
threshold fitting, the corresponding accepted yes/no/uncertain counts are
18/6/0, 22/6/1, and 21/5/3. The target is 80% relevance among decisive calibration
pairs with at least five accepted examples. Actual held-out-anchor relevance is
75.0%, 78.6%, and 80.8%. Thresholds are model-specific and are not calibrated
probabilities. Qwen returns matches for nine anchors and withholds all matches
for six. No model shows a large, independently validated advantage.

The selection rule chose Qwen as the only configuration meeting the selective
precision target with sufficient support. On its family vectors, HDBSCAN and
average-linkage hierarchy settings were compared without dimensionality reduction.
The exploratory hierarchy choice is distance cut 0.097619 with minimum three
families per retained group: 44 groups cover 196 families. HDBSCAN with minimum
three families forms 27 groups covering 159 families. The hierarchy coassigns
19 known relevant pairs, no known irrelevant pairs, and one uncertain pair,
recovering 52.8% of known related pairs. Settings were selected on development
data, so these scores do not establish final quality.

All 44 complete group memberships were assistant-reviewed. Thirty are
provisionally coherent, 13 mixed, and one uncertain. This review finds errors
missed by the judged-pair benchmark, including `libffi` grouped with cryptographic
libraries. Three user reference checks pass; ruff/uv and OpenSSL/OpenVPN remain
unassigned and are not counted as successful separation.

The implementation is in `evaluate_families.py`; local reports are
[FAMILY_COMPARISON.md](clustering_runs/baseline/FAMILY_COMPARISON.md),
[GROUPING_COMPARISON.md](clustering_runs/baseline/GROUPING_COMPARISON.md), and
[GROUP_REVIEW.md](clustering_runs/baseline/GROUP_REVIEW.md). Group labels combine
class-based TF-IDF keywords with separately recorded assistant-proposed names.
Original inputs, user judgments, and control-model vectors remain unchanged.
Cached reruns, distinct-family rankings, normalized Qwen vectors, family totals,
and reviewed membership hashes were checked. No candidate memberships have
been published or promoted into `discovery.py`.
