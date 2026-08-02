import os
from pathlib import Path
from dotenv import load_dotenv

# Load backend/.env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://recompro_user:1234@localhost:5432/recompro",
)

EXCEL_PATH = os.getenv("EXCEL_PATH", "../products.xlsx")

BASE_MODEL = os.getenv("BASE_MODEL", "intfloat/multilingual-e5-base")
DENSE_MODEL_DIR = os.getenv("DENSE_MODEL_DIR", "").strip() or None

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "./artifacts")).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Allow multiple comma-separated frontend origins or fallback to common local dev ports
_origins_env = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176,http://localhost:3000",
)

FRONTEND_ORIGIN = [origin.strip() for origin in _origins_env.split(",") if origin.strip()]