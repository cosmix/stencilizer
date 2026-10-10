# Web UI hosting feasibility

Findings checked on 2026-09-25. This is an exploratory assessment; no browser
prototype, deployment, or performance benchmark has been completed.

## Conclusion

Free hosting on Cloudflare looks feasible if font processing runs in the user's
browser. Hosting the existing Python processing pipeline directly on free
Cloudflare Workers is a poor fit because of compute limits and runtime differences.

The recommended first experiment is a small Pyodide browser prototype that converts
a representative font. Its dependency compatibility, startup time, processing time,
and memory use should determine whether to pursue this architecture.

## Hosting options

| Approach | Hosting cost | Fit for Stencilizer |
| --- | --- | --- |
| Static UI on Cloudflare; processing in the browser | Free static hosting | Most promising option for a free service |
| UI plus Python processing in a Cloudflare Worker | Free tier available | Requires runtime changes; CPU budget is restrictive |
| Static UI plus Python container backend | Workers Paid plan starts at $5/month, plus applicable usage | Closest to the existing Python runtime |

Cloudflare Workers Static Assets serves static assets with free, unlimited
requests. Requests that execute Worker code have separate limits and billing.
See [static asset billing](https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/).

## Why free Workers is a poor backend fit

Stencilizer uses Python, FontTools, Pydantic, and a CPU-oriented glyph geometry
pipeline. The current processor dispatches glyph transformations through
`ProcessPoolExecutor` in `src/stencilizer/core/processor.py`. Configuring one worker
still creates a process pool; it does not select an in-process execution path.

The free Workers plan allows 10 ms of CPU time per invocation and 128 MB of memory.
CPU time is distinct from elapsed time spent waiting for network operations.
Whole-font processing is expected to exceed this CPU allowance, although this has
not been measured. See [Workers limits](https://developers.cloudflare.com/workers/platform/limits/).

Cloudflare supports Python through Pyodide/WebAssembly. This is not a conventional
Python server environment, and the existing process-pool execution path would need
replacement. Dependency compatibility would also need verification. See
[how Python Workers work](https://developers.cloudflare.com/workers/languages/python/how-python-workers-work/)
and [supported Python packages](https://developers.cloudflare.com/workers/languages/python/packages/).

Moving to paid Workers would provide more CPU allowance, but would still require
runtime adaptation. A Python container is a separate deployment option.

## Browser processing with Pyodide

The proposed application would have the following flow:

1. Cloudflare serves the UI, Python runtime assets, and application code.
2. The user selects a local font.
3. Pyodide runs the Python processing code in a browser Web Worker.
4. The UI displays progress and glyph previews and exposes bridge settings.
5. The browser offers the generated font as a download.

A browser Web Worker runs computation away from the UI thread. It is distinct from
a Cloudflare Worker: processing uses the user's device rather than Cloudflare's
server execution budget. See [Pyodide Web Worker integration](https://pyodide.org/en/stable/usage/webworker.html).

This design could reuse much of the Python geometry implementation. Pyodide's
package index includes FontTools, Pydantic, and pydantic-core, but their presence
does not establish compatibility with this project's required versions. See the
[Pyodide package index](https://index.pyodide.org/314.0.2).

### Required adaptations

- Add a sequential glyph-processing path that does not create Python processes.
- Adapt font input and output to browser-provided bytes or Pyodide's virtual
  filesystem, then return output bytes for download.
- Expose a small processing API independent of the CLI.
- Adapt file logging and progress reporting for the browser UI.
- Validate all required dependencies against a selected Pyodide release.

Pyodide has limitations around threading and multiprocessing; browser Web Workers
do not automatically make `ProcessPoolExecutor` available. Sequential processing
inside one Web Worker is the simplest initial experiment. See
[Pyodide's threading and multiprocessing FAQ](https://pyodide.org/en/stable/usage/faq.html#can-i-use-threading-multiprocessing-subprocess).

### Benefits and tradeoffs

Font files can stay on the user's device. The conversion flow would require no
upload endpoint, server-side font storage, or backend compute service. Runtime and
package assets still need to be downloaded.

The main unknowns are initial download and startup time, memory consumption, and
conversion speed on slower devices. Running work off the UI thread helps
responsiveness but does not reduce the total computation required. Large fonts and
complex outlines need explicit evaluation.

## Container fallback

A conventional Python container could expose the existing pipeline through an
HTTP API while Cloudflare serves the UI. This preserves more of the current runtime
and process model, although request handling, upload limits, concurrency, progress,
and temporary-file cleanup would still need implementation.

Cloudflare Containers requires the Workers Paid plan, starting at $5/month. The
plan includes some container usage; additional resource usage and associated
services can incur charges. It is not an entirely free option. See
[Container pricing](https://developers.cloudflare.com/containers/platform/pricing/).

## Suggested feasibility test

Before building the full UI:

1. Load the processing dependencies in a browser Web Worker using Pyodide.
2. Run a complete font conversion through a sequential processing path.
3. Compare the generated font with native Python output for glyph correctness.
4. Measure cold startup, subsequent startup, conversion time, and peak memory
   where browser tooling permits.
5. Repeat with representative small and large fonts and a slower target device.
6. Check progress updates, cancellation behavior, and output download.

Proceed with static hosting if compatibility and performance are acceptable.
Otherwise, use these measurements to evaluate a container backend. No hosting
architecture has been committed to yet.
