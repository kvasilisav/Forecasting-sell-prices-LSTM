import logging

import hydra
import numpy as np
import pandas as pd
import torch
from dataloader import DataProcessor, TimeSeriesDataset
from dvc_utils import download_data
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from model import LSTMModel

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def predict(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    download_data(cfg)

    processor = DataProcessor(cfg)
    ts = processor._load_sales_data()
    data = processor.preprocess_data(ts)

    dataset = TimeSeriesDataset(data, cfg.data.seq_length)
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size)

    model = LSTMModel.load_from_checkpoint(
        cfg.infer.model_path, model_cfg=cfg.model, train_cfg=cfg.train
    )
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in loader:
            y_hat = model(x)
            predictions.append(y_hat.cpu().numpy())
            actuals.append(y.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    predictions = processor.y_scaler.inverse_transform(predictions)
    actuals = processor.y_scaler.inverse_transform(actuals)

    dates = pd.date_range(
        start=cfg.infer.start, periods=len(predictions) + cfg.data.seq_length + 1
    )[cfg.data.seq_length + 1 : len(predictions) + cfg.data.seq_length + 1]

    result = pd.DataFrame(
        {
            "date": dates,
            "prediction": predictions.flatten(),
            "actual": actuals.flatten(),
        }
    )

    result.to_csv(cfg.infer.output_path, index=False)
    logger.info(f"Predictions saved to {cfg.infer.output_path}")


if __name__ == "__main__":
    predict()
