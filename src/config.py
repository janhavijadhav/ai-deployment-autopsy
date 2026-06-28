"""Central settings — loaded from environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"

    # Vector store
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "procurement_contracts"

    # Cache
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600
    SEMANTIC_CACHE_THRESHOLD: float = 0.92

    # Database
    DUCKDB_PATH: str = "data/procurement.duckdb"
    SQLITE_PATH: str = "data/sap_mirror.db"

    # Auth
    SUPPLIER_API_BASE_URL: str = "http://localhost:8001"
    OAUTH2_TOKEN_URL: str = "http://localhost:8001/oauth/token"
    OAUTH2_CLIENT_ID: str = "procurement-agent"
    OAUTH2_CLIENT_SECRET: str = ""
    OAUTH2_SCOPE: str = "supplier.read contract.read approval.write"

    # Observability
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"
    PROMETHEUS_PORT: int = 9091

    # Schema monitor
    SCHEMA_MONITOR_ENABLED: bool = True
    SCHEMA_SNAPSHOT_PATH: str = "data/schema_snapshots/"
    SCHEMA_ALERT_WEBHOOK: str = ""

    # Reranker (cross-encoder — third RAG stage)
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_K: int = 5                  # Final results returned after reranking
    RERANKER_CANDIDATE_MULTIPLIER: int = 4   # Fetch this many × top_k before reranking

    # Evals
    EVAL_FAITHFULNESS_THRESHOLD: float = 0.85
    EVAL_RELEVANCY_THRESHOLD: float = 0.80
    ADVERSARIAL_EVAL_COUNT: int = 50


settings = Settings()
