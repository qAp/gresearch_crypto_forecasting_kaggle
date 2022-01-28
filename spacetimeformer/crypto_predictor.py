

import os, sys
import numpy as np
import pandas as pd
import torch

import spacetimeformer as stf
from spacetimeformer.train import create_model
from spacetimeformer.data.crypto.config import (
    DIR_BASE, DIR_PREPROCESS, ASSET_IDS, TIME_FEATURES,
    MIN_YEAR_SCALING, MAX_YEAR_SCALING)
from spacetimeformer.data.crypto.utils import unstack_assetid
from spacetimeformer.data.crypto import CryptoTimeSeries, CryptoDataset



class CryptoPredictor:
    def __init__(self, args=None):
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._create_timeseries()
        self._create_forecaster()

    def _create_timeseries(self):
        data_path = f'{DIR_PREPROCESS}/train_tindex.feather'
        target_cols = [f'Target_{id}' for id in ASSET_IDS] + self.args.xtra_target_cols
        feature_cols = []

        val_split = .2
        test_split = .15
        null_value = -999

        self.ts = CryptoTimeSeries(data_path, target_cols, feature_cols,
                                   val_split, test_split, null_value)

        self.target_cols = target_cols
        self.feature_cols = feature_cols
        self.val_split = val_split
        self.test_split = test_split
        self.null_value = null_value
        self.x_dim = len(TIME_FEATURES) + len(feature_cols)
        self.y_dim = len(target_cols)

    def _create_forecaster(self):
        forecaster = create_model(self.args, self.x_dim, self.y_dim)
        forecaster.set_inv_scaler(self.ts.reverse_scaling)
        forecaster.set_null_value(self.null_value)
        forecaster.eval()
        self.forecaster = forecaster
        self.forecaster.to(self.device)

    def initialize_context(self,
                           csv_path=f'{DIR_BASE}/supplemental_train.csv',
                           empty=False):

        columns = ['timestamp'] + self.feature_cols + self.target_cols

        df_supp = pd.read_csv(csv_path)
        df_supp = unstack_assetid(df_supp)
        df_supp = df_supp[columns]
        df_supp.sort_values('timestamp', axis=0, ascending=True, inplace=True)
        df_supp = df_supp.tail(self.args.context_points)

        if empty:
            df = pd.DataFrame(columns=columns)
            df.loc[:, 'timestamp'] = df_supp['timestamp'].values
            self._context_df = df
        else:
            self._context_df = df_supp

    def _get_raw_df(self, test_df, sample_prediction_df):
        test_df = unstack_assetid(test_df)
        test_df = test_df[['timestamp', ] + self.feature_cols]

        raw_df = pd.concat([self._context_df, test_df], axis=0)
        return raw_df

    def _preprocess_features(self, raw_df):
        time_df = pd.to_datetime(raw_df['timestamp'], unit='s')
        df = stf.data.timefeatures.time_features(
            time_df,
            min_year=MIN_YEAR_SCALING, max_year=MAX_YEAR_SCALING,
            main_df=raw_df)

        df.replace(np.inf, np.nan, inplace=True)
        df.replace(-np.inf, np.nan, inplace=True)

        df = self.ts.apply_scaling(df)
        if self.ts._feature_scaler:
            df = self.ts.apply_feature_scaling(df)

        if self.ts.null_value is not None:
            df.fillna(self.ts.null_value, inplace=True)

        return df

    def frame2tensors(self, df):
        context_slice = df[:self.args.context_points]
        target_slice = df[self.args.context_points:
                        self.args.context_points + self.args.target_points + 1]

        xc = context_slice[TIME_FEATURES + self.feature_cols].values
        yc = context_slice[self.target_cols].values

        xt = target_slice[TIME_FEATURES + self.feature_cols].values
        yt = target_slice[self.target_cols].values

        xc = torch.from_numpy(xc).float()
        yc = torch.from_numpy(yc).float()
        xt = torch.from_numpy(xt).float()
        yt = torch.from_numpy(yt).float()
        return xc, yc, xt, yt

    def predict(self, xc, yc, xt, yt):
        xcb = xc[None, ...].to(self.device)
        ycb = yc[None, ...].to(self.device)
        xtb = xt[None, ...].to(self.device)
        ytb = yt[None, ...].to(self.device)

        with torch.no_grad():
            outputs, (logits, labels) = self.forecaster(xcb, ycb, xtb, ytb)
            pr = outputs.mean
            pr = pr.cpu().numpy()
            pr = self.forecaster._inv_scaler(pr)
            pr = pr.ravel()

        return pr

    def update_context(self, test_df, pr):
        target = pd.DataFrame(pr[None, ...], columns=self.target_cols)
        target['timestamp'] = test_df['timestamp'].unique().item()
        self._context_df = self._context_df.iloc[1:].append(
            target, ignore_index=True)
