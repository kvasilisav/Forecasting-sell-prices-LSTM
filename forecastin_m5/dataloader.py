import os
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.x, self.y = self._create_sequences(data, seq_length)
        
    def _create_sequences(self, data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length - 1):
            _x = data[i:(i + seq_length), :]
            _y = data[i + seq_length, 0]
            x.append(_x)
            y.append(_y)
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(np.array(y).reshape(-1, 1))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class DataProcessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.y_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.calendar = self._load_calendar()
        
    def _load_calendar(self):
        calendar = pd.read_csv(os.path.join(self.cfg.data.data_path, "calendar.csv"))
        calendar["date"] = pd.to_datetime(calendar["date"])
        return calendar[["date", "weekday"]]
    
    def _load_sales_data(self):
        sales = pd.read_csv(os.path.join(self.cfg.data.data_path, "sales_train_validation.csv"))
        item_store_id = sales.iloc[self.cfg.data.item_index]["id"]
        
        # Extract sales columns (d_1 to d_1913)
        sales_cols = [col for col in sales.columns if col.startswith('d_')]
        ts = sales[sales["id"] == item_store_id][sales_cols].T
        ts.columns = ["sales"]
        ts.index = pd.date_range(start="2011-01-29", periods=len(ts))
        ts = ts.reset_index().rename(columns={"index": "date"})  # Преобразуем индекс в столбец
        return ts
    
    def _add_lag_features(self, df):
        for i in self.cfg.data.lags:
            df[f"lag_{i}"] = df["sales"].shift(i)
        
        for i in self.cfg.data.rolling_windows:
            df[f"rolling_mean_{i}"] = df["sales"].shift(28).rolling(i).mean()
            df[f"rolling_std_{i}"] = df["sales"].shift(28).rolling(i).std()
        return df
    
    def _add_weekday_embeddings(self, df):
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(self.calendar, on="date", how="left")

        df["weekday"] = df["weekday"].fillna("Unknown")
        
        embeddings = self.cfg.data.weekday_embeddings
        
        for i in range(1, 5):
            df[f"wd{i}"] = 0
            for day, emb in embeddings.items():
                df.loc[df["weekday"] == day, f"wd{i}"] = emb[i-1]
                
        return df
    
    def preprocess_data(self, ts):
        ts = self._add_lag_features(ts)
        ts = self._add_weekday_embeddings(ts)

        features = ts[[
            "sales", "lag_7", "lag_1", "lag_28", "lag_365",
            "rolling_mean_7", "rolling_std_7",
            "rolling_mean_14", "rolling_std_14",
            "rolling_mean_28", "rolling_std_28",
            "rolling_mean_60", "rolling_std_60",
            "wd1", "wd2", "wd3", "wd4"
        ]]

        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        scaled_features = self.scaler.fit_transform(features)
        return scaled_features
    
    def load_and_preprocess(self):
        ts = self._load_sales_data()
        return self.preprocess_data(ts)