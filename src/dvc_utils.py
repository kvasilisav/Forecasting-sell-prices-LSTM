import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_dvc(cfg):
    """Initialize DVC with local storage using Hydra config"""

    if not Path(".dvc").exists():
        subprocess.run(["dvc", "init"], check=True)
        logger.info("DVC initialized")

        os.makedirs(cfg.data.data_path, exist_ok=True)
        os.makedirs(cfg.data.processed_path, exist_ok=True)

        subprocess.run(["dvc", "add", cfg.data.data_path], check=True)
        logger.info(f"Local DVC storage configured for {cfg.data.data_path}")


def commit_data(cfg):
    """Commit data changes to DVC using Hydra config"""
    try:
        dvc_file = Path(cfg.data.data_path).with_suffix(".dvc")

        subprocess.run(["dvc", "add", cfg.data.data_path], check=True)
        subprocess.run(["git", "add", str(dvc_file), ".gitignore"], check=True)
        logger.info(f"Data changes committed to DVC for {cfg.data.data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data commit failed: {e}")
        raise


def download_data(cfg):
    """Download data from DVC cache using Hydra config"""
    try:
        data_path = Path(cfg.data.data_path)

        if not any(data_path.iterdir()):
            subprocess.run(["dvc", "pull", str(data_path)], check=True)
            logger.info(f"Data downloaded from DVC cache to {data_path}")
        else:
            logger.info("Data already exists locally")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data download failed: {e}")
        raise


def upload_data(cfg):
    """Upload data to DVC cache using Hydra config"""
    try:
        subprocess.run(["dvc", "push", str(Path(cfg.data.data_path))], check=True)
        logger.info(f"Data uploaded to DVC cache from {cfg.data.data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data upload failed: {e}")
        raise
