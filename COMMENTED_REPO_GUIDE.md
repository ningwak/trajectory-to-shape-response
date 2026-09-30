# Commented Study Edition

This directory is a reading-oriented copy of `trajectory-to-shape-response`.
The executable statements are intentionally identical to the standard version.
Only comments, docstrings, and explanatory Markdown have been expanded.

## Suggested offline reading order

1. `ALGORITHMS_PSEUDOCODE.md` — see the entire data flow without Python syntax.
2. `src/reflow_analysis/pipeline.py` — see how modules are connected.
3. `io.py` and `preprocessing.py` — understand the input contract and causal
   kinematics.
4. `splitter_inference.py` — follow horizontal sections, Wasserstein response,
   exact K=2 route bands, and the effective trajectory separator.
5. `passages.py` — understand structured G0-to-G3 passage records and route
   classification.
6. `online_state.py` — understand what information is causal and suitable for a
   future decision agent.
7. `metrics.py` — understand what is deliberately post-action evaluation.
8. `plots.py` — connect numerical objects to the six figures.
9. `topology.py` — optional general-scene baseline, outside the main pipeline.
10. `tests/` — small synthetic examples of the most important contracts.

The most useful regression examples are the exact one-dimensional K-means
counterexample, the closed-loop case with a missing G3 between two G0 entries,
the interpolated path endpoints, pooled passage-time aggregation, and map
matching after a node table has been reindexed.

## How to read the annotations

- **Why comments** explain a modelling or algorithm-selection reason.
- **Mechanism comments** explain exactly what the following code does.
- **Validity comments** distinguish a missing/invalid estimate from a physical
  zero.
- **Causality comments** identify where only past and current observations are
  used.
- **Scope comments** identify functionality that is optional or deliberately
  limited to this two-condition interview dataset.

The comments do not claim that every threshold is universal. Thresholds are
centralized in configuration dataclasses so they are visible, reproducible, and
easy to replace when more experimental runs become available.

## Logic-equivalence check

The study edition is checked against the standard repository by parsing both
sets of Python files, removing docstring expressions, and comparing the
remaining abstract syntax trees. This is stronger than a visual diff for this
purpose: comments do not enter the syntax tree, and changed executable syntax
would make the comparison fail.
