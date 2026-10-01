# Third-Party Notices

## Memobase

This artifact includes a patch derived from Memobase source code.

- Upstream project: <https://github.com/memodb-io/memobase>
- Upstream commit: `358c16bbc6d687937d79bc2f984a11c3be8da901`
- Upstream version: `v0.0.42`
- License: Apache License 2.0
- Distributed patch: `patches/memobase-v0.0.42-reasoning-models.patch`

The anonymous artifact authors modified the following upstream files:

- `src/server/api/memobase_server/llms/__init__.py`
- `src/server/api/memobase_server/llms/openai_model_llm.py`

The patch adjusts the startup LLM sanity check and adds compatibility handling
for the reasoning models used in the experiments. The complete Apache License
2.0 text applying to this Memobase-derived material is provided in
`licenses/memobase-APACHE-2.0.txt`.

The presence of this third-party material does not change the license applied
to the repository's original code, which remains specified by the top-level
`LICENSE` file.
