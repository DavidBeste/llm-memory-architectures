# CIMemories integrated paper artifacts

Open `REPORT.html` for the integrated visual report. This bundle is offline and makes no model calls.

- `response_analysis/` preserves the complete legacy factorial response export.
- `efficiency_analysis/` contains architecture-level macro-means and the underlying model-persona pipeline measurements.
- `memory_analysis/` contains direct pre/post-rerank memory availability, paired architecture/model contrasts, reranking effects, and difference-in-differences.
- `cross_stage_analysis/` joins returned-memory availability to response exposure.
- `figures/` and `includes/` contain generated SVG and LaTeX aids for paper preparation.
- `integrated_artifact_inputs.json` locks every selected input for exact regeneration.

Response metrics use one post-rerank generation per context (the lowest saved repeat index); pre-rerank and direct-memory metrics are unchanged.
