from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT = 10


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


@dataclass(frozen=True)
class LettaConfig:
    """
    Central config for Letta server interaction.

    You can override defaults via env vars:
      - LETTA_BASE_URL
      - LETTA_AGENT_NAME
      - LETTA_AGENT_MODEL
      - LETTA_AGENT_MODEL_ENDPOINT_TYPE
      - LETTA_AGENT_MODEL_ENDPOINT
      - LETTA_AGENT_REASONING_EFFORT
      - LETTA_AGENT_CONTEXT_WINDOW
      - LETTA_EMBEDDING_ENDPOINT_TYPE
      - LETTA_EMBEDDING_ENDPOINT
      - LETTA_EMBEDDING_MODEL
      - LETTA_EMBEDDING_DIM
      - LETTA_ARCHIVAL_SEARCH_LIMIT
      - LETTA_RERANK_MODE
      - LETTA_RERANK_CANDIDATE_SOURCE
      - LETTA_RERANK_CANDIDATE_LIMIT
      - LETTA_RERANK_OUTPUT_LIMIT
      - LETTA_RERANK_MODEL
      - LETTA_RERANK_BASE_URL
      - LETTA_RERANK_API_STYLE
      - LETTA_RERANK_API_KEY
      - LETTA_RERANK_TIMEOUT
      - LETTA_RERANK_MAX_TOKENS
      - LETTA_RERANK_REASONING_EFFORT
      - LETTA_RERANK_CHECKPOINT_DIR
      - LETTA_RERANK_PROMPT_MODE
      - VLLM_BASE_URL
      - VLLM_MODEL
      - GRAPHITI_NEO4J_URI
      - GRAPHITI_NEO4J_USER
      - GRAPHITI_NEO4J_PASSWORD
      - GRAPHITI_GROUP_PREFIX
      - GRAPHITI_SEARCH_LIMIT
      - GRAPHITI_INGEST_LIMIT
      - GRAPHITI_BATCH_SIZE
      - GRAPHITI_LLM_BASE_URL
      - GRAPHITI_LLM_API_KEY
      - GRAPHITI_LLM_MODEL
      - GRAPHITI_LLM_REASONING_EFFORT
      - GRAPHITI_EMBEDDING_BASE_URL
      - GRAPHITI_EMBEDDING_API_KEY
      - GRAPHITI_EMBEDDING_MODEL
      - GRAPHITI_EMBEDDING_DIM
      - LETTA_ATTACKER_RAG_FILE
      - LETTA_ATTACKER_RAG_SEARCH_LIMIT
      - MEMOBASE_PROJECT_URL
      - MEMOBASE_API_KEY
      - MEMOBASE_CONTEXT_MAX_TOKEN_SIZE
      - MEMOBASE_SERVER_CONFIG_PATH
    """
    base_url: str = os.getenv("LETTA_BASE_URL", "http://localhost:8283/v1")
    agent_name: str = os.getenv("LETTA_AGENT_NAME", "memory-chat")
    agent_model: str = os.getenv("LETTA_AGENT_MODEL") or os.getenv("VLLM_MODEL") or "gpt-5.2"
    agent_model_endpoint_type: str = os.getenv("LETTA_AGENT_MODEL_ENDPOINT_TYPE", "openai")
    agent_model_endpoint: str = (
        os.getenv("LETTA_AGENT_MODEL_ENDPOINT")
        or os.getenv("VLLM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    agent_reasoning_effort: str | None = os.getenv("LETTA_AGENT_REASONING_EFFORT") or None
    agent_context_window: int = _env_int("LETTA_AGENT_CONTEXT_WINDOW", 128000)
    embedding_endpoint_type: str = os.getenv("LETTA_EMBEDDING_ENDPOINT_TYPE", "openai")
    embedding_endpoint: str = os.getenv("LETTA_EMBEDDING_ENDPOINT", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("LETTA_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = _env_int("LETTA_EMBEDDING_DIM", 1536)
    archival_search_limit: int = _env_int(
        "LETTA_ARCHIVAL_SEARCH_LIMIT",
        DEFAULT_ARCHIVAL_MEMORY_SEARCH_LIMIT,
    )
    rerank_mode: str = os.getenv("LETTA_RERANK_MODE", "lexical")
    rerank_candidate_source: str = os.getenv("LETTA_RERANK_CANDIDATE_SOURCE", "search")
    rerank_candidate_limit: int = _env_int("LETTA_RERANK_CANDIDATE_LIMIT", 200)
    rerank_output_limit: int = _env_int("LETTA_RERANK_OUTPUT_LIMIT", 20)
    rerank_model: str = os.getenv("LETTA_RERANK_MODEL") or os.getenv("VLLM_MODEL") or "gpt-5.2"
    rerank_prompt_mode: str = os.getenv("LETTA_RERANK_PROMPT_MODE", "legacy").strip().lower()
    graphiti_neo4j_uri: str = os.getenv("GRAPHITI_NEO4J_URI", "bolt://localhost:7687")
    graphiti_neo4j_user: str = os.getenv("GRAPHITI_NEO4J_USER", "neo4j")
    graphiti_neo4j_password: str = os.getenv("GRAPHITI_NEO4J_PASSWORD", "password")
    graphiti_group_prefix: str = os.getenv("GRAPHITI_GROUP_PREFIX", "letta-research-chat")
    graphiti_search_limit: int = _env_int("GRAPHITI_SEARCH_LIMIT", 20)
    graphiti_ingest_limit: int | None = _env_optional_int("GRAPHITI_INGEST_LIMIT")
    graphiti_batch_size: int = _env_int("GRAPHITI_BATCH_SIZE", 12)
    graphiti_llm_base_url: str | None = os.getenv("GRAPHITI_LLM_BASE_URL") or os.getenv("VLLM_BASE_URL")
    graphiti_llm_api_key: str | None = os.getenv("GRAPHITI_LLM_API_KEY") or os.getenv("VLLM_API_KEY")
    graphiti_llm_model: str | None = os.getenv("GRAPHITI_LLM_MODEL") or os.getenv("VLLM_MODEL")
    graphiti_llm_reasoning_effort: str | None = (
        os.getenv("GRAPHITI_LLM_REASONING_EFFORT")
        or os.getenv("LETTA_AGENT_REASONING_EFFORT")
        or None
    )
    graphiti_llm_structured_output_mode: str = os.getenv("GRAPHITI_LLM_STRUCTURED_OUTPUT_MODE", "json_object")
    graphiti_llm_max_tokens: int = _env_int("GRAPHITI_LLM_MAX_TOKENS", 4096)
    graphiti_embedding_base_url: str | None = os.getenv("GRAPHITI_EMBEDDING_BASE_URL") or os.getenv("LETTA_EMBEDDING_ENDPOINT")
    graphiti_embedding_api_key: str | None = os.getenv("GRAPHITI_EMBEDDING_API_KEY") or os.getenv("VLLM_API_KEY")
    graphiti_embedding_model: str | None = os.getenv("GRAPHITI_EMBEDDING_MODEL") or os.getenv("LETTA_EMBEDDING_MODEL")
    graphiti_embedding_dim: int = _env_int("GRAPHITI_EMBEDDING_DIM", _env_int("LETTA_EMBEDDING_DIM", 1024))
    graphiti_reranker_base_url: str | None = os.getenv("GRAPHITI_RERANKER_BASE_URL") or os.getenv("GRAPHITI_LLM_BASE_URL") or os.getenv("VLLM_BASE_URL")
    graphiti_reranker_api_key: str | None = os.getenv("GRAPHITI_RERANKER_API_KEY") or os.getenv("VLLM_API_KEY")
    graphiti_reranker_model: str | None = os.getenv("GRAPHITI_RERANKER_MODEL") or os.getenv("GRAPHITI_LLM_MODEL") or os.getenv("VLLM_MODEL")
    attacker_rag_file: str = os.getenv(
        "LETTA_ATTACKER_RAG_FILE",
        os.path.join("research_outputs", "attacker_rag_content.txt"),
    )
    attacker_rag_search_limit: int = _env_int("LETTA_ATTACKER_RAG_SEARCH_LIMIT", 5)
    memobase_project_url: str = os.getenv("MEMOBASE_PROJECT_URL", "http://localhost:8019")
    memobase_api_key: str = os.getenv("MEMOBASE_API_KEY", "secret")
    memobase_context_max_token_size: int = _env_int("MEMOBASE_CONTEXT_MAX_TOKEN_SIZE", 1000)
    memobase_server_config_path: str | None = os.getenv("MEMOBASE_SERVER_CONFIG_PATH") or None

    # Optional default persona file (for /init_archival shortcut)
    default_persona_file: str | None = None

    # History file settings (CLI UX)
    history_file: str = os.path.join(os.path.expanduser("~"), ".letta_chat_history")
    history_len: int = 2000
