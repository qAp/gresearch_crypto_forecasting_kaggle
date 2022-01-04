

import numpy as np
import pandas as pd


DIR_BASE = '/kaggle/input/g-research-crypto-forecasting/'

df = pd.read_csv(f'{DIR_BASE}/asset_details.csv')
ASSET_IDS = sorted(df['Asset_ID'].unique())

