import pytorch_lightning as pl
import torch
import torch.nn as nn

from .config import model_config, training_config


class LSTMModel(pl.LightningModule):
    """LSTM model for time series forecasting"""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            batch_first=True,
            dropout=model_config.dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(model_config.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        h0 = torch.zeros(
            model_config.num_layers, x.size(0), model_config.hidden_size
        ).to(x.device)

        c0 = torch.zeros(
            model_config.num_layers, x.size(0), model_config.hidden_size
        ).to(x.device)

        _, (hn, _) = self.lstm(x, (h0, c0))

        final_state = hn[-1]
        return self.fc(final_state)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=model_config.learning_rate,
            weight_decay=model_config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=training_config.patience,
            factor=0.5,
            min_lr=training_config.min_lr,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    import fire

    fire.Fire(LSTMModel)
