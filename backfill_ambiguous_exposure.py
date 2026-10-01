#!/usr/bin/env python3
"""Backfill ambiguous-exposure metrics and refresh existing evaluation tables.

The default mode is read-only. Pass --apply to update referenced
privacy_metrics_cimemories.json files. Add --regenerate-tables to replace
existing comparison Markdown files with freshly rendered C | L | A tables.
No model calls are made.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parent
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from letta_research_chat.cli import (  # noqa: E402
    compare_privacy_pipeline_cimemories_multi_reports,
    compare_privacy_pipeline_cimemories_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ambiguous exposure from saved per-repeat judgments. "
            "Defaults to a read-only dry run over OpenAI pipeline results."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="research_outputs",
        help="evaluation root to scan (default: research_outputs)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically add/update ambiguous and unknown exposure fields in metric JSON files",
    )
    parser.add_argument(
        "--regenerate-tables",
        action="store_true",
        help="regenerate and replace existing comparison Markdown files (requires --apply)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="include every pipeline; by default only agent_model values beginning with gpt- are backfilled",
    )
    parser.add_argument(
        "--quiet-reports",
        action="store_true",
        help="suppress verbose terminal tables while regenerating Markdown",
    )
    parser.add_argument(
        "--backup-dir",
        default=str(WORKSPACE / "backfill_backups"),
        help=(
            "directory for compressed pre-change backups "
            "(default: <repository>/backfill_backups)"
        ),
    )
    return parser.parse_args()


def resolve_stored_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else WORKSPACE / path


def exposure_stats(attributes: set[str], counts: Counter[str], repeat_count: int) -> list[dict[str, Any]]:
    return [
        {
            "attribute": attribute,
            "exposure_count": counts.get(attribute, 0),
            "repeat_count": repeat_count,
            "exposure_rate": counts.get(attribute, 0) / repeat_count,
            "exposure_percent": (counts.get(attribute, 0) / repeat_count) * 100,
        }
        for attribute in sorted(attributes)
    ]


def compute_backfill(payload: dict[str, Any]) -> dict[str, Any]:
    necessary = payload.get("necessary_attributes")
    inappropriate = payload.get("inappropriate_attributes")
    ambiguous = payload.get("ambiguous_attributes")
    repeated = payload.get("repeated_exposed_records")
    if not isinstance(necessary, list) or not all(isinstance(item, str) for item in necessary):
        raise ValueError("missing valid necessary_attributes")
    if not isinstance(inappropriate, list) or not all(isinstance(item, str) for item in inappropriate):
        raise ValueError("missing valid inappropriate_attributes")
    if not isinstance(ambiguous, list) or not all(isinstance(item, str) for item in ambiguous):
        raise ValueError("missing valid ambiguous_attributes")
    counts: Counter[str] = Counter()
    if isinstance(repeated, list) and repeated:
        for index, record in enumerate(repeated):
            if not isinstance(record, dict) or not isinstance(record.get("exposed_attributes"), dict):
                raise ValueError(f"repeat {index} is missing exposed_attributes")
            counts.update(record["exposed_attributes"].keys())
        repeat_count = len(repeated)
    else:
        exposed = payload.get("exposed_attributes")
        if not isinstance(exposed, dict):
            raise ValueError("missing repeated_exposed_records and exposed_attributes")
        counts.update(exposed.keys())
        repeat_count = 1
    necessary_set = set(necessary)
    inappropriate_set = set(inappropriate)
    ambiguous_set = set(ambiguous)
    exposed_set = set(counts)
    exposed_ambiguous = sorted(exposed_set & ambiguous_set)
    unknown_exposed = sorted(exposed_set - necessary_set - inappropriate_set - ambiguous_set)
    ambiguous_events = sum(counts.get(attribute, 0) for attribute in ambiguous_set)
    unknown_events = sum(counts.get(attribute, 0) for attribute in unknown_exposed)
    ambiguous_total = len(ambiguous_set)

    return {
        "metrics": {
            "ambiguous_total": ambiguous_total,
            "exposed_ambiguous_count": len(exposed_ambiguous),
            "average_exposed_ambiguous_count": ambiguous_events / repeat_count,
            "ambiguous_exposure_rate": (
                ambiguous_events / (ambiguous_total * repeat_count) if ambiguous_total else None
            ),
            "unknown_exposed_count": len(unknown_exposed),
            "average_unknown_exposed_count": unknown_events / repeat_count,
        },
        "ambiguous_attribute_exposure_percentages": exposure_stats(
            ambiguous_set,
            counts,
            repeat_count,
        ),
        "unknown_attribute_exposure_percentages": exposure_stats(
            set(unknown_exposed),
            counts,
            repeat_count,
        ),
        "exposed_ambiguous_attributes": exposed_ambiguous,
        "unknown_exposed_attributes": unknown_exposed,
    }


def updated_payload(payload: dict[str, Any], backfill: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        result["metrics"] = metrics
    metrics.update(backfill["metrics"])
    for key, value in backfill.items():
        if key != "metrics":
            result[key] = value
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def create_backup_archive(paths: list[Path], backup_dir: Path) -> Path:
    """Archive original files plus a restoration manifest outside the scan root."""
    unique_paths = sorted({path.resolve() for path in paths})
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S_%fZ")
    archive_path = backup_dir / f"ambiguous_exposure_{timestamp}.tar.gz"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
        dir=backup_dir,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    manifest: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "files": [],
    }
    try:
        with tarfile.open(temporary_path, "w:gz") as archive:
            for index, path in enumerate(unique_paths, start=1):
                if not path.is_file():
                    raise FileNotFoundError(f"backup source not found: {path}")
                member = f"files/{index:04d}_{path.name}"
                archive.add(path, arcname=member, recursive=False)
                manifest["files"].append(
                    {
                        "archive_member": member,
                        "original_path": str(path),
                    }
                )
            manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            manifest_info = tarfile.TarInfo("manifest.json")
            manifest_info.size = len(manifest_bytes)
            manifest_info.mtime = int(dt.datetime.now(dt.UTC).timestamp())
            archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        os.replace(temporary_path, archive_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return archive_path


def discover_metric_files(root: Path, all_models: bool) -> tuple[list[Path], list[str]]:
    metric_paths: set[Path] = set()
    errors: list[str] = []
    for pipeline_path in sorted(root.rglob("privacy_pipeline_cimemories.json")):
        try:
            summary = json.loads(pipeline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{pipeline_path}: cannot read pipeline summary: {exc}")
            continue
        model = summary.get("agent_model") if isinstance(summary, dict) else None
        if not all_models and (not isinstance(model, str) or not model.startswith("gpt-")):
            continue
        contexts = summary.get("contexts") if isinstance(summary, dict) else None
        if not isinstance(contexts, list):
            errors.append(f"{pipeline_path}: missing contexts list")
            continue
        for context in contexts:
            raw_path = context.get("privacy_metrics_cimemories_json") if isinstance(context, dict) else None
            if not isinstance(raw_path, str):
                errors.append(f"{pipeline_path}: context missing metrics path")
                continue
            metric_path = resolve_stored_path(raw_path).resolve()
            if not metric_path.is_file():
                errors.append(f"{pipeline_path}: metrics file not found: {metric_path}")
                continue
            metric_paths.add(metric_path)
    return sorted(metric_paths), errors


def extract_pipeline_paths(markdown: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"`([^`]*privacy-pipeline-cimemories[^`]*)`", markdown):
        paths.append(match.group(1).strip())
    for line in markdown.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        for cell in line.strip().strip("|").split("|"):
            candidate = cell.strip().strip("`")
            if "privacy-pipeline-cimemories" in candidate and not any(
                character.isspace() for character in candidate
            ):
                paths.append(candidate)
    return list(dict.fromkeys(paths))


def discover_comparison_tables(root: Path) -> tuple[list[tuple[Path, list[str]]], list[str]]:
    tables: list[tuple[Path, list[str]]] = []
    errors: list[str] = []
    patterns = (
        "privacy_pipeline_cimemories_comparison_slack_*.md",
        "privacy_pipeline_cimemories_multi_comparison_slack_*.md",
        "privacy_pipeline_cimemories_multi_summary_slack_*.md",
    )
    seen: set[Path] = set()
    for pattern in patterns:
        for markdown_path in sorted(root.rglob(pattern)):
            resolved_markdown = markdown_path.resolve()
            if resolved_markdown in seen:
                continue
            seen.add(resolved_markdown)
            try:
                paths = extract_pipeline_paths(markdown_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"{markdown_path}: cannot read comparison: {exc}")
                continue
            valid_paths = [path for path in paths if resolve_stored_path(path).exists()]
            if len(valid_paths) < 2:
                errors.append(f"{markdown_path}: could not resolve at least two pipeline paths")
                continue
            tables.append((resolved_markdown, valid_paths))
    return tables, errors


def regenerate_table(markdown_path: Path, pipeline_paths: list[str], quiet: bool) -> Path:
    output: io.StringIO | None = io.StringIO() if quiet else None
    redirect = contextlib.redirect_stdout(output) if output is not None else contextlib.nullcontext()
    with redirect:
        if "multi_" in markdown_path.name or len(pipeline_paths) > 2:
            generated = compare_privacy_pipeline_cimemories_multi_reports(pipeline_paths)
        else:
            generated = compare_privacy_pipeline_cimemories_reports(
                pipeline_paths[0],
                pipeline_paths[1],
            )
    generated_path = resolve_stored_path(str(generated)).resolve()
    if generated_path != markdown_path:
        os.replace(generated_path, markdown_path)
    return markdown_path


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = WORKSPACE / root
    root = root.resolve()
    if not root.is_dir():
        print(f"error: evaluation root not found: {root}", file=sys.stderr)
        return 2
    if args.regenerate_tables and not args.apply:
        print("error: --regenerate-tables requires --apply", file=sys.stderr)
        return 2

    os.chdir(WORKSPACE)
    metric_paths, discovery_errors = discover_metric_files(root, args.all_models)
    changed: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    errors = list(discovery_errors)
    for metric_path in metric_paths:
        try:
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("top-level JSON value is not an object")
            backfill = compute_backfill(payload)
            new_payload = updated_payload(payload, backfill)
            if new_payload != payload:
                changed.append((metric_path, new_payload, backfill["metrics"]))
        except Exception as exc:
            errors.append(f"{metric_path}: {exc}")

    tables: list[tuple[Path, list[str]]] = []
    if args.regenerate_tables:
        tables, table_errors = discover_comparison_tables(root)
        errors.extend(table_errors)

    if args.apply and errors:
        for error in errors:
            print(f"warning: {error}", file=sys.stderr)
        print(
            f"summary: mode=apply-refused pipelines_metric_files={len(metric_paths)} "
            f"changed={len(changed)} tables_regenerated=0 errors={len(errors)}"
        )
        return 1

    if args.apply and (changed or tables):
        backup_dir = Path(args.backup_dir).expanduser().resolve()
        try:
            backup_path = create_backup_archive(
                [metric_path for metric_path, _, _ in changed]
                + [markdown_path for markdown_path, _ in tables],
                backup_dir,
            )
        except Exception as exc:
            print(f"error: backup failed; no evaluation files were changed: {exc}", file=sys.stderr)
            return 1
        print(f"BACKUP\t{backup_path}")

    action = "UPDATE" if args.apply else "WOULD UPDATE"
    for metric_path, new_payload, metrics in changed:
        relative = metric_path.relative_to(WORKSPACE) if metric_path.is_relative_to(WORKSPACE) else metric_path
        print(
            f"{action}\t{relative}\t"
            f"ambiguous={metrics['average_exposed_ambiguous_count']:.2f}/"
            f"{metrics['ambiguous_total']} ({(metrics['ambiguous_exposure_rate'] or 0) * 100:.2f}%)\t"
            f"unknown={metrics['average_unknown_exposed_count']:.2f}"
        )
        if args.apply:
            atomic_write_json(metric_path, new_payload)

    table_count = 0
    if args.regenerate_tables:
        for markdown_path, pipeline_paths in tables:
            try:
                regenerate_table(markdown_path, pipeline_paths, args.quiet_reports)
                table_count += 1
                relative = (
                    markdown_path.relative_to(WORKSPACE)
                    if markdown_path.is_relative_to(WORKSPACE)
                    else markdown_path
                )
                print(f"REGENERATED\t{relative}\truns={len(pipeline_paths)}")
            except Exception as exc:
                errors.append(f"{markdown_path}: regeneration failed: {exc}")

    for error in errors:
        print(f"warning: {error}", file=sys.stderr)
    mode = "applied" if args.apply else "dry-run"
    print(
        f"summary: mode={mode} pipelines_metric_files={len(metric_paths)} "
        f"changed={len(changed)} tables_regenerated={table_count} errors={len(errors)}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
