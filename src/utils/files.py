import os


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
