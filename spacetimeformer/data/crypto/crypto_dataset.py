
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import spacetimeformer as stf
from spacetimeformer.data import CSVTimeSeries, CSVTorchDset



class CryptoTimeSeries(CSVTimeSeries):
    
    def __init__(self, data_path, target_cols, feature_cols=None, 
                 val_split=.2, test_split=.15,
                 null_value=None):
        
        assert os.path.exists(data_path)
        self.data_path = data_path
        self.target_cols = target_cols
        self.feature_cols = [] if feature_cols is None else feature_cols
        self.val_split = val_split
        self.test_split = test_split
        self.null_value = null_value
        
        columns = ['timestamp',] + self.feature_cols + self.target_cols
        raw_df = pd.read_feather(data_path, columns=columns)
        
        time_df = pd.to_datetime(raw_df['timestamp'], unit='s')
        df = stf.data.timefeatures.time_features(time_df, raw_df)

        df.replace(np.inf, np.nan, inplace=True)
        df.replace(-np.inf, np.nan, inplace=True)

        self._make_splits(df, val_split, test_split)
        self._fit_scalers()
        self._scale_data()
        self._fillna()
        
    def _make_splits(self, df, val_split, test_split):
        # Train/Val/Test Split using holdout approach #

        def mask_intervals(mask, intervals, cond):
            for (interval_low, interval_high) in intervals:
                if interval_low is None:
                    interval_low = df["Datetime"].iloc[0].year
                if interval_high is None:
                    interval_high = df["Datetime"].iloc[-1].year
                mask[
                    (df["Datetime"] >= interval_low) & (df["Datetime"] <= interval_high)
                ] = cond
            return mask

        print('test_split', test_split)
        print('df', df.head())
        test_cutoff = len(df) - round(test_split * len(df))
        val_cutoff = test_cutoff - round(val_split * len(df))

        val_interval_low = df['Datetime'].iloc[val_cutoff]
        val_interval_high = df['Datetime'].iloc[test_cutoff - 1]
        val_intervals = [(val_interval_low, val_interval_high)]

        test_interval_low = df['Datetime'].iloc[test_cutoff]
        test_interval_high = df['Datetime'].iloc[-1]
        test_intervals = [(test_interval_low, test_interval_high)]

        train_mask = df["Datetime"] > pd.Timestamp.min
        val_mask = df["Datetime"] > pd.Timestamp.max
        test_mask = df["Datetime"] > pd.Timestamp.max
        train_mask = mask_intervals(train_mask, test_intervals, False)
        train_mask = mask_intervals(train_mask, val_intervals, False)
        val_mask = mask_intervals(val_mask, val_intervals, True)
        test_mask = mask_intervals(test_mask, test_intervals, True)

        if (train_mask == False).all():
            print(f"No training data detected for file {self.data_path}")
            
        self._train_data = df[train_mask]
        self._val_data = df[val_mask]
        self._test_data = df[test_mask]
        
    def _fit_scalers(self):
        self._scaler = StandardScaler()
        self._scaler.fit(self._train_data[self.target_cols].values)

        if self.feature_cols:
            self._feature_scaler = StandardScaler()
            self._feature_scaler.fit(self._train_data[self.feature_cols].values)
        else:
            self._feature_scaler = None
        
    def apply_feature_scaling(self, df):
        scaled = df.copy(deep=True)
        scaled[self.feature_cols] = self._feature_scaler.transform(
            df[self.feature_cols].values
        )
        return scaled
    
    def reverse_feature_scaling_df(self, df):
        scaled = df.copy(deep=True)
        scaled[self.feature_cols] = self._feature_scaler.inverse_transform(
            df[self.feature_cols].values
        )
        return scaled
    
    def reverse_feature_scaling(self, array):
        return self._feature_scaler.inverse_transform(array)
    
    def _scale_data(self):
        self._train_data = self.apply_scaling(self._train_data)
        self._val_data = self.apply_scaling(self._val_data)
        self._test_data = self.apply_scaling(self._test_data)

        if self._feature_scaler:
            self._train_data = self.apply_feature_scaling(self._train_data)
            self._val_data = self.apply_feature_scaling(self._val_data)
            self._test_data = self.apply_feature_scaling(self._test_data)
        
    def _fillna(self):
        if self.null_value is not None:
            self._train_data.fillna(self.null_value, inplace=True)
            self._val_data.fillna(self.null_value, inplace=True)
            self._test_data.fillna(self.null_value, inplace=True)       


class CryptoDataset(CSVTorchDset):
    def __init__(self,        
                 crypto_time_series: CryptoTimeSeries,
                 split: str = "train",
                 context_points: int = 128,
                 target_points: int = 32,
                 time_resolution: int = 1):
        
        super().__init__(crypto_time_series, split, 
                         context_points, target_points, time_resolution)
        
    def __getitem__(self, i):
        start = self._slice_start_points[i]
        series_slice = self.series.get_slice(
            self.split,
            start=start,
            stop=start
            + self.time_resolution * (self.context_points + self.target_points),
            skip=self.time_resolution,
        ).drop(columns=["Datetime"])
        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :],
        )
        ctxt_x = ctxt_slice[
            ['Year', 'Month', 'Day', 'Weekday', 'Hour', 'Minute'] + 
            self.series.feature_cols].values
        ctxt_y = ctxt_slice[self.series.target_cols].values

        trgt_x = trgt_slice[
            ['Year', 'Month', 'Day', 'Weekday', 'Hour', 'Minute'] + 
            self.series.feature_cols].values
        trgt_y = trgt_slice[self.series.target_cols].values

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)    