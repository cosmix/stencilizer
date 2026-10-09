<!-- generated automatically on knowledge writes — do not edit by hand -->

# Knowledge Index

> Read this index first, then only what it points to: the section for your area in a tier-1 summary (`rg -n '^## ' <file>` lists them) and the tier-2 topics your task touches. A specific question is cheaper to pull than to read — `loom knowledge context --query "..."` returns the matching sections quoted.

## Tier 1 — Summaries

| File | Description | Lines |
| --- | --- | --- |
| [architecture.md](architecture.md) | High-level component relationships, data flow, module dependencies | 42 |
| [entry-points.md](entry-points.md) | Key files agents should read first | 25 |
| [patterns.md](patterns.md) | Architectural patterns discovered in the codebase | 26 |
| [conventions.md](conventions.md) | Coding conventions discovered in the codebase | 57 |
| [mistakes.md](mistakes.md) | Mistakes made and lessons learned - what to avoid | 235 |
| [stack.md](stack.md) | Dependencies, frameworks, and tooling used in the project | 26 |
| [concerns.md](concerns.md) | Technical debt, warnings, and issues to address | 86 |

## Tier 2 — Topics

### architecture

| Topic | Blurb | Lines |
| --- | --- | --- |
| [gui](architecture/gui.md) | GUI package layout, threading model, save safety | 92 |

### patterns

| Topic | Blurb | Lines |
| --- | --- | --- |
| [bridge-algorithm](patterns/bridge-algorithm.md) | Island detection, bridge placement, contour surgery, multi-island cases | 71 |
| [variable-replay](patterns/variable-replay.md) | Variable-font stencil replay, validation, delta solve, rounding, measured rates | 43 |

### mistakes

| Topic | Blurb | Lines |
| --- | --- | --- |
| [ci-release](mistakes/ci-release.md) | setup-uv lacks major tags; version bumps need uv lock | 33 |
| [gui](mistakes/gui.md) | fontTools TTFont iteration, Qt widget lifetime in tests | 16 |
| [offscreen-screenshots](mistakes/offscreen-screenshots.md) | Offscreen 2x GUI grabs need a large screen config file | 10 |
| [readme-layout](mistakes/readme-layout.md) | Place README screenshots beside the content they illustrate | 13 |
| [review-and-completion-gates](mistakes/review-and-completion-gates.md) | Malformed reviews, fingerprint drift, IV process traps, loom tool quirks | 59 |
| [variable-fonts](mistakes/variable-fonts.md) | Pool start-method side effects, CFF2 delta checks, analyzer blind spots | 31 |
