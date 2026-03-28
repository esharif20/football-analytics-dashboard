import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    LOCAL_DEV_MODE: bool = os.getenv("LOCAL_DEV_MODE", "true").lower() == "true"
    LOCAL_STORAGE_DIR: str = os.getenv("LOCAL_STORAGE_DIR", "./uploads")
    OWNER_OPEN_ID: str = os.getenv("OWNER_OPEN_ID", "local-dev-user")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret")
    FORGE_API_URL: str = os.getenv("BUILT_IN_FORGE_API_URL", "")
    FORGE_API_KEY: str = os.getenv("BUILT_IN_FORGE_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")


settings = Settings()

import logging as _logging

_logger = _logging.getLogger("api.config")

# QUAL-04 / D-07: Refuse to start with insecure JWT in non-dev mode
if settings.JWT_SECRET == "dev-secret" and not settings.LOCAL_DEV_MODE:
    raise ValueError(
        "JWT_SECRET is set to the insecure default 'dev-secret' but LOCAL_DEV_MODE is not enabled. "
        "Set a strong JWT_SECRET for production or set LOCAL_DEV_MODE=true for development."
    )
