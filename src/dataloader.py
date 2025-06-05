import os
import re

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
        for idx in range(len(data) - seq_length - 1):
            x.append(data[idx : (idx + seq_length), :])
            y.append(data[idx + seq_length, 0])
        return torch.FloatTensor(np.array(x)), torch.FloatTensor(
            np.array(y).reshape(-1, 1)
        )

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
        calendar["day_of_week"] = calendar["date"].dt.dayofweek
        calendar["day_of_month"] = calendar["date"].dt.day
        calendar["month"] = calendar["date"].dt.month
        calendar["year"] = calendar["date"].dt.year
        calendar["is_event"] = calendar["event_name_1"].notna().astype(int)
        return calendar[
            ["d", "date", "day_of_week", "day_of_month", "month", "year", "is_event"]
        ]

    def _load_sales_data(self):
        sales = pd.read_csv(
            os.path.join(self.cfg.data.data_path, "sales_train_validation.csv")
        )
        sales["item_store_id"] = sales["item_id"] + "_" + sales["store_id"]
        item_store_id = sales.iloc[self.cfg.data.item_index]["item_store_id"]

        col_pattern = re.compile(r"^d_\d+$")
        sales_data = sales[sales["item_store_id"] == item_store_id].loc[
            :, sales.columns.str.match(col_pattern)
        ]
        date_range = pd.date_range(start="2011-01-29", periods=len(sales_data.columns))
        ts = pd.DataFrame(
            {"date": date_range, "sales": sales_data.values.flatten()}
        ).set_index("date")
        return ts

    def _add_time_features(self, df):
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["year"] = df.index.year

        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        return df

    def _add_lag_features(self, df):
        lag_periods = [1, 7, 14, 28, 56, 365]
        for period in lag_periods:
            df[f"lag_{period}"] = df["sales"].shift(period)

        window_sizes = [7, 14, 28, 60, 90, 180]
        for window in window_sizes:
            df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
            df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

        return df

    def _merge_calendar_features(self, df):
        df = df.reset_index()
        df = (
            df.merge(self.calendar[["date", "is_event"]], on="date", how="left")
            .fillna(0)
            .set_index("date")
        )
        return df

    def preprocess_data(self, ts):
        ts = self._add_time_features(ts)
        ts = self._add_lag_features(ts)
        ts = self._merge_calendar_features(ts)

        ts = ts.replace([np.inf, -np.inf], np.nan)
        ts = ts.fillna(0)

        features = ts.drop(columns=["sales"])
        scaled_features = self.scaler.fit_transform(features)
        scaled_target = self.y_scaler.fit_transform(ts[["sales"]].values)

        return np.concatenate([scaled_target, scaled_features], axis=1)

    def load_and_preprocess(self):
        ts = self._load_sales_data()
        return self.preprocess_data(ts)
