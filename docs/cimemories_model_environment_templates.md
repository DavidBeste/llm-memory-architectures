# CIMemories model environment templates

The three model configurations are stored as non-secret, sourceable files:

| Evaluation family | Environment template | Safe launcher |
|---|---|---|
| GPT-5.6-sol | `experiment_templates/cimemories_gpt_5_6_sol.env` | `scripts/start_cimemories_gpt_5_6_sol.sh` |
| GLM-5.3-Flash | `experiment_templates/cimemories_glm_5_3.env` | `scripts/start_cimemories_glm_5_3.sh` |
| DeepSeek-V4-Flash | `experiment_templates/cimemories_deepseek_v4.env` | `scripts/start_cimemories_deepseek_v4.sh` |

Export `OPENAI_API_KEY` and `TOGETHER_API_KEY` first. The launchers validate
that both exist, source the selected template, hide credential values in their
status output, and start `letta-chat` in the repository root. For example:

```bash
scripts/start_cimemories_glm_5_3.sh
```

To configure the current shell persistently instead of immediately launching
the chat client, source exactly one template:

```bash
source experiment_templates/cimemories_gpt_5_6_sol.env
```

All three templates keep the response-exposure judge on GPT-5.2 and the direct
returned-memory semantic judge on `gpt-6-sol` through OpenAI. The latter is used
only by `/compute_memory_stage_metrics` for graph and profile memory; list
memory uses deterministic exact matching. Model-specific response generation,
reranking, and Graphiti settings remain faithful to their experiment family.

Memobase is a separate service and does not inherit these shell variables.
Before a profile-memory run, ensure its `config.yaml` contains the intended
internal model configuration and restart `memobase-server-api`. Changing the
shell template alone switches list-memory, Graphiti, reranking, response
generation, and judges, but not Memobase's internal extraction model.
Sanitized templates for the three reported configurations are stored in
`artifact_configs/memobase/`. The pinned upstream revision and reasoning-model
compatibility patch are documented in `ARTIFACT.md`.
