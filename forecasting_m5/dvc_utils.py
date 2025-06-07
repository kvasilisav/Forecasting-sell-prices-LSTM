import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def download_data(cfg):
    """Download data from dvc cache"""
    try:
        data_path = Path(cfg.data.data_path)
        os.makedirs(cfg.data.data_path, exist_ok=True)
        os.makedirs(cfg.data.processed_path, exist_ok=True)
        if not any(data_path.iterdir()):
            subprocess.run(["dvc", "pull"], check=True)
            logger.info(f"Data downloaded from DVC cache to {data_path}")
        else:
            logger.info("Data already exists locally")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data download failed: {e}")
        raise
