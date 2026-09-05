"""Environment / credentials helpers."""
import os
import sys
from typing import Optional

from peka import logger


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        logger.error(f"Required environment variable '{name}' is not set. "
                     f"Copy peka/.env.example to peka/.env and fill in.")
        sys.exit(1)
    return val


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def hf_login(required: bool = False) -> None:
    """Authenticate to HuggingFace using HF_TOKEN. Required for HEST1K + H-optimus-0."""
    token = get_env("HF_TOKEN")
    if not token:
        msg = "HF_TOKEN not set — gated downloads (HEST1K, H-optimus-0, UNI) will fail."
        if required:
            logger.error(msg)
            sys.exit(1)
        logger.warning(msg)
        return
    from huggingface_hub import login
    login(token=token)
    logger.info("HuggingFace login: OK")


def wandb_setup(api_key: Optional[str] = None) -> Optional[str]:
    """Set WANDB_API_KEY in the environment. Returns the key (or None if missing)."""
    if api_key is None:
        api_key = get_env("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    return api_key
