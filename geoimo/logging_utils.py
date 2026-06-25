import sys
import logging
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: str = "INFO",
                  log_mode: str = "w") -> logging.Logger:
    logger = logging.getLogger("geoimo")
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(numeric_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, mode=log_mode)
        fh.setLevel(numeric_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_every(logger: logging.Logger, i: int, step: int, msg: str) -> None:
    if step <= 1 or (i % step) == 0:
        logger.info(msg)
