"""Runtime configuration, loaded from environment variables or a local .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # A blank line such as DATA_DIR= in .env means "use the default", not Path("").
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    # Provider API keys. A provider with no key is skipped by every chain.
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    hf_token: str = ""
    github_models_token: str = ""

    # Model chains, as comma-separated provider:model references tried in order. When a
    # model's daily budget is exhausted the chain moves to the next one and stays there.
    # The answer chain starts with the smallest model that calls tools reliably; the
    # judge chain starts with the cheapest capable Gemini model, flash-lite.
    answer_chain: str = (
        "groq:openai/gpt-oss-20b,groq:openai/gpt-oss-120b,"
        "openrouter:nvidia/nemotron-3.5-lightning:free,"
        "openrouter:nvidia/nemotron-3-super-120b-a12b:free,"
        "gemini:gemini-3.1-flash-lite,gemini:gemini-flash-lite-latest,gemini:gemini-3.5-flash-lite"
    )
    reflect_chain: str = (
        "gemini:gemini-3.1-flash-lite,gemini:gemini-flash-lite-latest,gemini:gemini-3.5-flash-lite,"
        "openrouter:nvidia/nemotron-3-super-120b-a12b:free,groq:openai/gpt-oss-120b,"
        "groq:openai/gpt-oss-20b"
    )
    # gemini-3.6-flash is not in this chain: its free tier allows 20 requests per day.
    # The hf and github providers are registered but not in these chains: hf's free
    # monthly inference credit is exhausted within a handful of calls, and GitHub Models
    # answered every call with HTTP 410 (scheduled retirement brownout) when tested.
    judge_chain: str = (
        "gemini:gemini-3.5-flash-lite,gemini:gemini-3.1-flash-lite,gemini:gemini-flash-lite-latest,"
        "gemini:gemini-3.5-flash,gemini:gemini-3.7-flash,gemini:gemini-3.8-flash,"
        "groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b,"
        "openrouter:nvidia/nemotron-3-super-120b-a12b:free"
    )

    # gpt-oss models spend completion tokens on hidden reasoning before answering.
    # Low effort is enough for grounded extraction and keeps latency and cost down.
    groq_reasoning_effort: str = "low"

    # Groq's free tier is rate limited, so judge calls run at low concurrency and
    # retry with backoff rather than failing the whole evaluation run.
    eval_concurrency: int = Field(default=2, ge=1, le=16)
    eval_max_retries: int = Field(default=8, ge=0, le=20)
    eval_max_wait_seconds: float = Field(default=90.0, ge=1.0)
    # RAGAS prompts carry long few-shot demonstrations that double the token cost of
    # every judge call. Dropping them keeps the metric definitions and output parsing
    # intact while fitting the run inside the free-tier daily budget.
    eval_lean_prompts: bool = True

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # torch (sentence-transformers) for development and CI; onnx in the deployed image,
    # which drops torch and keeps the container inside a 512 MB free-tier instance.
    embedding_backend: Literal["torch", "onnx"] = "torch"
    # CPU is deliberate: the model is tiny, results are identical everywhere the project
    # runs, and the MPS backend segfaulted when embeddings ran inside the async judge.
    embedding_device: str = "cpu"

    qdrant_url: str = "http://localhost:16333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "quickcommerce_kb"
    # When set, Qdrant runs embedded in-process against this directory instead of a
    # server. The deployed container uses it so the image needs no external services.
    qdrant_path: str = ""

    # MySQL database
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "quickcommerce"

    # MySQL backs the tools in local development.
    # SQLite remains available for tests.
    tools_backend: Literal["mysql", "sqlite"] = "mysql"
    sqlite_path: Path = PROJECT_ROOT / "data" / "darkstore.sqlite"
    tool_call_cap: int = Field(default=3, ge=1, le=10)

    # The agentic loop is the default. Retrieval-only mode is kept so the baseline
    # can be reproduced against the same knowledge base.
    agent_mode: Literal["retrieval", "agentic"] = "agentic"
    # Upper bound on model turns in one question, independent of the tool cap.
    agent_max_steps: int = Field(default=6, ge=2, le=12)

    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    chunk_tokens: int = Field(default=220, ge=48, le=2048)
    chunk_overlap_tokens: int = Field(default=40, ge=0, le=512)

    # Retries on 429 and 5xx for the answer model, waiting as long as Groq asks, up to
    # this many seconds. A longer wait means a daily budget is gone; stop and resume later.
    llm_max_retries: int = Field(default=6, ge=0, le=20)
    llm_max_wait_seconds: float = Field(default=120.0, ge=1.0)

    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=700, ge=64, le=8192)

    data_dir: Path = PROJECT_ROOT / "data"
    # Built single-page app served at "/". When the directory is absent, "/" returns the
    # same JSON as /meta so the API stays usable without a frontend build.
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"
    # Committed evaluation reports, surfaced read-only through /meta.
    eval_results_dir: Path = PROJECT_ROOT / "eval" / "results"

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"

    @property
    def catalog_dir(self) -> Path:
        return self.data_dir / "catalog"

    def api_keys(self) -> dict[str, str]:
        return {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "openrouter": self.openrouter_api_key,
            "hf": self.hf_token,
            "github": self.github_models_token,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
