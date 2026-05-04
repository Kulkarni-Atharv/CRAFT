"""Centralised logging setup — call get_logger() in every module."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured (e.g. multiple imports)
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
