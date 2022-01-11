
import os, sys
import numpy as np
import pandas as pd

from spacetimeformer.data.crypto.config import DIR_BASE, ASSET_IDS, FEATURES
from spacetimeformer.data.crypto.utils import unstack_assetid


train = pd.read_csv(f'{DIR_BASE}/train.csv')

train.to_feather('/kaggle/working/train.feather')

# Make each row a unique timestamp
df = unstack_assetid(train)

df = df.sort_values('timestamp', axis=0)
val_pct = .2
val_cut = int((1 - val_pct) * len(df))
df.loc[df.index < val_cut, 'split'] = 'train'
df.loc[df.index >= val_cut, 'split'] = 'valid'

fn = '/kaggle/working/train_tindex.feather'
df.to_feather(fn)

feature_cols = [f'{feature}_{id}' for id in ASSET_IDS for feature in FEATURES]

is_train = df['split'] == 'train'

stats_train = df[feature_cols][is_train].describe()

# VWAP_10 has `inf` values. Removing these and recalculate basic stats
srs = df.loc[is_train, 'VWAP_10']
srs = srs[(- np.inf < srs) & (srs < np.inf)]
stats_train.loc[:, 'VWAP_10'] = srs.describe()

(
    stats_train
    .reset_index()
    .to_csv('/kaggle/working/stats_train.csv', index=False)
    )


