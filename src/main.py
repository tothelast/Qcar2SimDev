#!/usr/bin/env python3
"""
Entry point for SimLingo-Qcar2 minimal integration demo.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repository root is on sys.path so `src` is importable
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add simlingo directory to path for simlingo_training imports
SIMLINGO_DIR = REPO_ROOT / "simlingo"
if str(SIMLINGO_DIR) not in sys.path:
    sys.path.insert(0, str(SIMLINGO_DIR))

from src.integration.main_bridge import run_cli


def setup_logging(debug: bool = False) -> Path:
    """
    Configure logging with file output and console output.

    Args:
        debug: If True, set file logging to DEBUG level. Otherwise INFO level.

    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"simlingo_qcar2_{timestamp}.log"

    # Determine log levels
    file_level = logging.DEBUG if debug else logging.INFO
    console_level = logging.INFO

    # Create formatters
    file_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Remove any existing handlers
    root_logger.handlers.clear()

    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler (minimal logging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers_modules").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: file={log_file.name}, file_level={logging.getLevelName(file_level)}, console_level={logging.getLevelName(console_level)}")

    return log_file


def main():
    p = argparse.ArgumentParser(description="Run SimLingo-Qcar2 integration demo")
    p.add_argument("--hz", type=float, default=5.0, help="Control loop frequency (default: 5 Hz)")
    p.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    p.add_argument("--show_agents_comments", action="store_true", help="Show model's reasoning/commentary")
    p.add_argument("--show_current_instruction", action="store_true", help="Show current instruction")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG level logging to file")
    args = p.parse_args()

    # Setup logging
    log_file = setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("SimLingo-QCar2 Integration Starting")
    logger.info(f"Configuration: hz={args.hz}, duration={args.duration}s, debug={args.debug}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    try:
        return run_cli(
            hz=args.hz,
            duration=args.duration,
            show_agents_comments=args.show_agents_comments,
            show_current_instruction=args.show_current_instruction
        )
    except Exception:
        logger.exception("Fatal error in main execution")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

