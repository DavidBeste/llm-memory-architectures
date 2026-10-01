# Memobase experiment configurations

These templates capture the non-secret Memobase server configuration used by
the three CIMemories model families. Memobase does not inherit the shell
variables used by `letta-chat`; copy the appropriate template to the pinned
Memobase checkout and replace the two credential placeholders locally.

```bash
cp artifact_configs/memobase/deepseek-v4-flash.config.yaml.template \
  "$MEMOBASE_ROOT/src/server/api/config.yaml"
```

Never commit the rendered file. The templates intentionally retain model,
embedding, token-budget, and profile-taxonomy settings because each can change
retrieved profile content. The server checkout used for the reported runs was
Memobase `v0.0.42` at commit
`358c16bbc6d687937d79bc2f984a11c3be8da901`, with
`patches/memobase-v0.0.42-reasoning-models.patch` applied.

The GPT response-model experiment used Memobase's established `gpt-4o-mini`
profile extraction configuration. The GLM and DeepSeek experiments used their
respective Together-hosted model for Memobase profile extraction. All three
used `text-embedding-3-small` at 1536 dimensions inside Memobase.
