import os
from dataclasses import dataclass

import torch


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""

    data_path: str = os.path.join("..", "input", "m5-forecasting-accuracy")
    seq_length: int = 28
    train_ratio: float = 0.8
    batch_size: int = 32
    target_column: str = "sales"
    item_index: int = 6780


@dataclass
class ModelConfig:
    """Configuration for model architecture"""

    input_size: int = 18
    hidden_size: int = 512
    num_layers: int = 4
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class TrainingConfig:
    """Configuration for training process"""

    max_epochs: int = 500
    patience: int = 50
    min_lr: float = 1e-7
    gpus: int = 1 if torch.cuda.is_available() else 0
    checkpoint_dir: str = "checkpoints"
    checkpoint_filename: str = "best_model"


@dataclass
class InferenceConfig:
    """Configuration for inference"""

    model_path: str = os.path.join("checkpoints", "best_model.ckpt")
    output_path: str = "predictions.csv"


data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
