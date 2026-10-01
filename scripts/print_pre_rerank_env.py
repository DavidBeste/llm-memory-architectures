#!/usr/bin/env python3
"""Print the effective pre-rerank replay environment without exposing secrets."""

from __future__ import annotations

import os


def effective(name: str, default: str) -> tuple[str, str]:
    value = os.getenv(name)
    return (value, "environment") if value else (default, "default")


def print_value(name: str, default: str) -> str:
    value, source = effective(name, default)
    print(f"{name:<40} {value:<36} [{source}]")
    return value


def print_secret(name: str) -> None:
    status = "<set>" if os.getenv(name) else "<unset>"
    print(f"{name:<40} {status}")


def main() -> None:
    print("Pre-rerank replay environment")
    print("=" * 78)
    print("\nLetta and response generation")
    print_value("LETTA_BASE_URL", "http://localhost:8283/v1")
    print_value("LETTA_SERVER_TIMEOUT", "120")
    print_value("LETTA_AGENT_MODEL_ENDPOINT_TYPE", "openai")
    print_value("LETTA_AGENT_MODEL_ENDPOINT", "https://api.openai.com/v1")
    print_value("LETTA_AGENT_CONTEXT_WINDOW", "128000")
    print_value("LETTA_EMBEDDING_ENDPOINT_TYPE", "openai")
    print_value("LETTA_EMBEDDING_ENDPOINT", "https://api.openai.com/v1")
    print_value("LETTA_EMBEDDING_MODEL", "text-embedding-3-small")
    print_value("LETTA_EMBEDDING_DIM", "1536")
    print("NOTE: generation model and reasoning are loaded from the saved run")
    print("      (expected: gpt-5.6-sol, medium). Embeddings are not used by replay.")

    print("\nExposure judge")
    judge_base = print_value("OPENAI_BASE_URL", "https://api.openai.com/v1")
    judge_style = print_value("OPENAI_API_STYLE", "responses")
    judge_model = print_value("LETTA_JUDGE_MODEL", os.getenv("VLLM_MODEL") or "gpt-5.2")
    judge_effort = print_value("LETTA_JUDGE_REASONING_EFFORT", "<unset>")
    judge_tokens = print_value("LETTA_JUDGE_MAX_TOKENS", "2048")
    print_value("LETTA_RESEARCH_OPENAI_TIMEOUT", "180")

    print("\nConcurrency")
    print_value("LETTA_RESEARCH_QUERY_CONCURRENCY", "4")
    print_value("LETTA_RESEARCH_OPENAI_CONCURRENCY", "8")

    print("\nEndpoint overrides")
    print_value("VLLM_BASE_URL", "<unset>")
    print_value("VLLM_MODEL", "<unset>")
    print_value("OLLAMA_BASE_URL", "<unset>")

    print("\nCredentials (values deliberately hidden)")
    for name in (
        "OPENAI_API_KEY",
        "VLLM_API_KEY",
        "OLLAMA_API_KEY",
    ):
        print_secret(name)

    warnings: list[str] = []
    if judge_base != "https://api.openai.com/v1":
        warnings.append("OPENAI_BASE_URL differs from the existing replay checkpoint.")
    if judge_style != "responses":
        warnings.append("OPENAI_API_STYLE should be 'responses' for the existing checkpoint.")
    if judge_model != "gpt-5.2":
        warnings.append("LETTA_JUDGE_MODEL should resolve to 'gpt-5.2'.")
    if judge_effort != "none":
        warnings.append("LETTA_JUDGE_REASONING_EFFORT should be 'none'.")
    if judge_tokens != "4096":
        warnings.append("LETTA_JUDGE_MAX_TOKENS should be '4096'.")
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY is unset; the OpenAI judge cannot authenticate.")
    if os.getenv("VLLM_BASE_URL") or os.getenv("VLLM_MODEL") or os.getenv("VLLM_API_KEY"):
        warnings.append("VLLM variables are set and may redirect or reconfigure the judge.")
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_KEY"):
        warnings.append("Ollama variables are set and may redirect or reconfigure the judge.")

    print("\nCompatibility check")
    if warnings:
        for warning in warnings:
            print(f"[WARN] {warning}")
    else:
        print("[OK] Environment matches the existing pre-rerank checkpoint settings.")
    print("[INFO] The Letta container's internal generation key cannot be read from this shell.")


if __name__ == "__main__":
    main()
