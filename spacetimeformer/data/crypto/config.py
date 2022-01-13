

import numpy as np
import pandas as pd


DIR_BASE = '/kaggle/input/g-research-crypto-forecasting'
DIR_PREPROCESS = '/kaggle/input/stfa01-data-preprocessing'

FEATURES = ['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 
            'Target']

TIME_FEATURES = ['Year', 'Month', 'Day', 'Weekday', 'Hour', 'Minute']

df = pd.read_csv(f'{DIR_BASE}/asset_details.csv')
ASSET_IDS = sorted(df['Asset_ID'].unique())

