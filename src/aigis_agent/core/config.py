"""Centralized runtime settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = SRC_DIR.parent


class Settings(BaseSettings):
    """App configuration for API, GIS providers and local sqlite storage."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AIGIS_",
        extra="ignore",
    )

    app_name: str = "AIGIS Agent"
    app_version: str = "0.1.0"
    api_prefix: str = "/v1"

    host: str = "0.0.0.0"
    port: int = 8000

    amap_api_key: str | None = None
    amap_base_url: str = "https://restapi.amap.com"
    amap_timeout_s: float = 6.0
    amap_retry_count: int = 1

    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    nominatim_timeout_s: float = 8.0
    nominatim_retry_count: int = 1
    nominatim_user_agent: str = "aigis-agent"
    osrm_base_url: str = "https://router.project-osrm.org"

    default_transport_profile: str = "driving"
    default_search_limit: int = 10

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    agent_query_max_length: int = 300

    poi_ai_refine_enabled: bool = True
    poi_ai_confidence_threshold: float = 0.72
    poi_ai_timeout_s: float = 8.0
    poi_ai_retry_zero_result: bool = True
    poi_ai_alpha_result_threshold: int = 3
    poi_ai_sparse_result_threshold: int = 2
    poi_ai_result_gain_threshold: float = 0.08
    poi_ai_query_max_length: int = 120
    poi_parallel_recall_enabled: bool = True
    poi_parallel_city_switch_margin: float = 0.08
    poi_parallel_city_switch_min_score: float = 0.62
    poi_ai_strong_landmark_enabled: bool = True
    poi_ai_strong_landmark_suggest_confidence_threshold: float = 0.62
    poi_ai_strong_landmark_confidence_threshold: float = 0.78
    poi_ai_strong_landmark_min_query_len: int = 2
    poi_ai_strong_landmark_max_query_len: int = 12

    sqlite_url: str = f"sqlite:///{WORKSPACE_DIR / 'data' / 'aigis.db'}"

    static_dir: Path = PACKAGE_DIR / "web" / "static"
    template_dir: Path = PACKAGE_DIR / "web" / "templates"


settings = Settings()
