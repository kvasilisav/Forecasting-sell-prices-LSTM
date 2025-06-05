import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from dataloader import DataProcessor, TimeSeriesDataset
from dvc_utils import download_data, setup_dvc
from model import LSTMModel

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    setup_dvc(cfg)
    download_data(cfg)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    processor = DataProcessor(cfg)
    ts = processor._load_sales_data()
    data = processor.preprocess_data(ts)

    dataset = TimeSeriesDataset(data, cfg.data.seq_length)
    train_size = int(cfg.data.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.data.batch_size)

    model = LSTMModel(cfg.model, cfg.train)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        filename=cfg.train.checkpoint_filename,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=cfg.train.patience, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
