# Anonymous CIMemories evaluation artifact

This directory is a non-destructive, sanitized `github`-scope copy of the locked paper pipelines and paper-report bundle. Source directories were not modified.

Operational identifiers, provider response identifiers, credentials and credential metadata, absolute user paths, execution fingerprints, exact run timestamps, and agent/conversation identifiers were removed or replaced. Paths inside copied files use portable `artifact://` or `repository://` references.

The apparent personal information retained in prompts, memories, and responses belongs to synthetic CIMemories personas. It is retained because it is the scientific object measured by the privacy evaluation; it is not human-subject or author information.

See `artifact_manifest.json` for the stable model/architecture/persona layout and `anonymization_report.json` for the automated release audit. Verify file integrity from the repository root with `python3 scripts/verify_anonymized_cimemories_artifact.py <this-directory>`.
