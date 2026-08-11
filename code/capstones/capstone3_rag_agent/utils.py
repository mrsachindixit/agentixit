import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return default
    return val


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
