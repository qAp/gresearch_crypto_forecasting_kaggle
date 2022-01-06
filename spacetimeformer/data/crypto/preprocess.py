
import os, sys
import numpy as np
import pandas as pd

from spacetimeformer.data.crypto.config import DIR_BASE
from spacetimeformer.data.crypto.utils import unstack_assetid


train = pd.read_csv(f'{DIR_BASE}/train.csv')

train.to_feather('/kaggle/working/train.feather')

df = unstack_assetid(train)

dt = pd.to_datetime(df['timestamp'], unit='s')
dt = dt.apply(lambda x: str(x))
dt = dt.str[:-3]  # "%Y-%m-%d %H:%M"
df['Datetime'] = dt.copy()
df.drop('timestamp', axis=1, inplace=True)

fn = '/kaggle/working/train_tindex.feather'
df.to_feather(fn)


