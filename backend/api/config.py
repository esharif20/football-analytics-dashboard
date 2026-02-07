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
