"""Consistent logger setup."""
import logging


def get_logger(name: str = "peka", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        log.addHandler(handler)
    log.setLevel(level)
    return log
