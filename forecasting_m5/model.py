import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.autograd import Variable


class LSTMModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.lstm = nn.LSTM(
            input_size=self.model_cfg.input_size,
            hidden_size=self.model_cfg.hidden_size,
            num_layers=self.model_cfg.num_layers,
            batch_first=True,
            dropout=model_cfg.dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(model_cfg.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(self.model_cfg.dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(self.model_cfg.dropout),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        h_1 = Variable(
            torch.zeros(
                self.model_cfg.num_layers, x.size(0), self.model_cfg.hidden_size
            )
        )

        c_1 = Variable(
            torch.zeros(
                self.model_cfg.num_layers, x.size(0), self.model_cfg.hidden_size
            )
        )

        _, (hn, cn) = self.lstm(x, (h_1, c_1))

        final_state = hn.view(
            self.model_cfg.num_layers, x.size(0), self.model_cfg.hidden_size
        )[-1]

        out = self.fc(final_state)

        return out

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

        mae = torch.mean(torch.abs(y_hat - y))
        rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.model_cfg.learning_rate,
            weight_decay=self.hparams.model_cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.train_cfg.patience,
            factor=self.hparams.train_cfg.factor,
            min_lr=self.hparams.train_cfg.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
