
import os, sys
import numpy as np
import pandas as pd

from spacetimeformer.data.crypto.config import DIR_BASE



def unstack_train_asset_id(df):
    '''
    Args:
        df (pd.DataFrame): Data from competition train.csv.
            
    Returns:
        df (pd.DataFrame): Asset IDs are in separate columns. 
            Each row should be a unique timestamp.
            Columns are '{feature}_{asset_id}'.
    '''
    df = df.pivot(index='timestamp', columns='Asset_ID')
    new_columns = [f'{feature}_{asset_id}' for feature, asset_id in df.columns.values]
    df.columns = new_columns
    df.reset_index(inplace=True)

    assert df['timestamp'].nunique() == len(df)
    return df


if __name__ == '__main__':

    train = pd.read_csv(f'{DIR_BASE}/train.csv')

    train.to_feather('/kaggle/working/train.feather')

    df = unstack_train_asset_id(train)
    fn = '/kaggle/working/train_tindex.feather'
    df.to_feather(fn)


