# Recommended direction for package discovery

The experiments now produce useful candidate categories, but automatic membership
still mixes software roles. Use the hierarchy to discover groups, then review
membership before those groups drive published trend charts.

## What the latest experiment established

The 460 described packages reduce to 437 families after collapsing `@version`
variants. Formula/cask roles and distinct canonical names remain separate.
Every model uses the same name-plus-description text. Qwen also receives one
fixed task instruction, applied symmetrically to every package.

| Model configuration | Relevant distinct-family neighbors / 75 | Relevant among decisive judgments |
| --- | ---: | ---: |
| MiniLM | 25 | 39.1% |
| MPNet | 27 | 42.2% |
| Qwen3-Embedding-0.6B with task instruction | 28 | 42.4% |

Qwen does not establish a large improvement over MPNet. Rejecting weak matches
helps more: leave-one-anchor-out threshold fitting returns 21 relevant, five
irrelevant, and three uncertain Qwen matches. Relevance is 80.8% among decisive
judgments, but only nine of 15 anchors receive a match. Including accepted
uncertain pairs gives relevance bounds of 72.4–82.8%. These are small development
results with assistant judgments, not independent validation or confidence scores.

## Grouping result

The declared selection rule chose Qwen for exploration. On its family vectors,
average-linkage hierarchical clustering gave a more useful conservative cut than
HDBSCAN in this experiment:

| Configuration | Groups | Assigned families | Related / unrelated / uncertain judged pairs within groups |
| --- | ---: | ---: | --- |
| HDBSCAN, minimum three families | 27 | 159 | 14 / 10 / 3 |
| Average linkage, distance cut 0.097619, minimum three families | 44 | 196 | 19 / 0 / 1 |

The hierarchy recovers 52.8% of the known related pairs. Its zero observed
unrelated coassignments cover a small sample and do not establish whole-group
accuracy. Complete membership inspection found **30 provisionally coherent
groups, 13 mixed groups, and one uncertain group**.

Useful candidates include media players, browsers, terminal emulators, 3D
modeling, container tools, and local Kubernetes tools. Mixed groups include
version and dependency managers, database servers and clients, and a
cryptographic-library group containing `libffi`.

Three user reference checks pass. The ruff/uv and OpenSSL/OpenVPN checks remain
unassigned, rather than counting as successful separation.

## Next implementation

1. Keep Qwen plus the hierarchy as an exploration candidate and MPNet as the
   cheaper control. Stop adding embedding models for now.
2. Start a draft category file from the 30 provisionally coherent groups. Give
   each a readable name and explicit package identities. Split or reject mixed
   groups; do not publish automatic membership directly.
3. Keep weak and uncertain matches unassigned. Use model-specific thresholds,
   report coverage, and describe withholding as insufficient evidence.
4. Check fresh package families after choosing the draft categories. Existing
   development judgments have already influenced selection, so reuse cannot
   establish independent final quality.

Refer to the local [family comparison](clustering_runs/baseline/FAMILY_COMPARISON.md),
[grouping comparison](clustering_runs/baseline/GROUPING_COMPARISON.md), and
[complete assistant group review](clustering_runs/baseline/GROUP_REVIEW.md).
The notebook and published charts retain their existing settings.
