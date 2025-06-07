import logging
import os

import git
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from dataloader import DataProcessor, TimeSeriesDataset
from dvc_utils import download_data
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split

from model import LSTMModel

logger = logging.getLogger(__name__)


def get_git_commit():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


class ValidationMetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.val_losses = []
        self.val_mae = []
        self.val_rmse = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        self.val_losses.append(metrics["val_loss"].item())

        if "val_mae" in metrics:
            self.val_mae.append(metrics["val_mae"].item())
        if "val_rmse" in metrics:
            self.val_rmse.append(metrics["val_rmse"].item())

    def on_train_end(self, trainer, pl_module):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = np.arange(1, len(self.val_losses) + 1)

        ax.plot(epochs, self.val_losses, label="Validation Loss")
        if self.val_mae:
            ax.plot(epochs, self.val_mae, label="Validation MAE")
        if self.val_rmse:
            ax.plot(epochs, self.val_rmse, label="Validation RMSE")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title("Validation Metrics During Training")
        ax.legend()
        ax.grid(True)

        trainer.logger.experiment.log_figure(
            run_id=trainer.logger.run_id,
            figure=fig,
            artifact_file="validation_metrics.png",
        )
        plt.close(fig)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    pl.seed_everything(cfg.train.seed, workers=True)

    mlflow_logger = MLFlowLogger(
        experiment_name="sales_forecasting", tracking_uri=cfg.infer.mlflow_uri
    )

    mlflow_logger.log_hyperparams(
        {"git_commit": get_git_commit(), **OmegaConf.to_container(cfg, resolve=True)}
    )

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    download_data(cfg)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    processor = DataProcessor(cfg)
    data = processor.load_and_preprocess()
    dataset = TimeSeriesDataset(data, cfg.data.seq_length)
    train_size = int(cfg.data.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_data, batch_size=cfg.data.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_data, batch_size=cfg.data.batch_size, num_workers=4)

    model = LSTMModel(cfg.model, cfg.train)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        filename=cfg.train.checkpoint_filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=cfg.train.patience, mode="min"
    )

    metrics_callback = ValidationMetricsCallback()

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=[checkpoint_callback, early_stopping, metrics_callback],
        logger=mlflow_logger,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    sample_input = next(iter(val_loader))[0][0].unsqueeze(0)
    torch.onnx.export(
        model,
        sample_input,
        os.path.join(cfg.train.checkpoint_dir, "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        },
    )

    mlflow_logger.experiment.log_artifact(
        run_id=mlflow_logger.run_id,
        local_path=os.path.join(cfg.train.checkpoint_dir, "model.onnx"),
    )

    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
    logger.info(f"ONNX model exported to {cfg.train.checkpoint_dir}/model.onnx")


if __name__ == "__main__":
    train()
