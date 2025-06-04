import fire
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import data_config, inference_config
from .dataloader import DataProcessor, TimeSeriesDataset
from .model import LSTMModel


def predict():
    """Make predictions using trained model"""
    processor = DataProcessor()
    ts = processor.load_data()
    data = processor.preprocess_data(ts)

    dataset = TimeSeriesDataset(data, data_config.seq_length)
    loader = DataLoader(dataset, batch_size=data_config.batch_size, shuffle=False)

    model = LSTMModel.load_from_checkpoint(inference_config.model_path)
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
        start="2011-01-29", periods=len(predictions) + data_config.seq_length + 1
    )[data_config.seq_length + 1 : len(predictions) + data_config.seq_length + 1]

    result = pd.DataFrame(
        {
            "date": dates,
            "prediction": predictions.flatten(),
            "actual": actuals.flatten(),
        }
    )

    result.to_csv(inference_config.output_path, index=False)
    print(f"Predictions saved to {inference_config.output_path}")

    return result


if __name__ == "__main__":
    fire.Fire(predict)
