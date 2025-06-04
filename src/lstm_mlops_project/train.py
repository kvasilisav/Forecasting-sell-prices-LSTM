import os

import fire
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from .config import data_config, training_config
from .dataloader import DataProcessor, TimeSeriesDataset
from .model import LSTMModel


def train():
    """Train the LSTM model"""
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    processor = DataProcessor()
    ts = processor.load_data()
    data = processor.preprocess_data(ts)

    dataset = TimeSeriesDataset(data, data_config.seq_length)

    train_size = int(data_config.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_data, batch_size=data_config.batch_size, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(val_data, batch_size=data_config.batch_size, num_workers=4)

    model = LSTMModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath=training_config.checkpoint_dir,
        filename=training_config.checkpoint_filename,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=training_config.patience * 2,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        gpus=training_config.gpus,
        enable_progress_bar=True,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    fire.Fire(train)
