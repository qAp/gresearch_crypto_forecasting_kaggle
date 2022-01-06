
import pandas as pd




def unstack_assetid(df):
    '''
    Unstack Asset_ID and make each row a unique timestamp.

    Args:
        df (pd.DataFrame): Data from competition. e.g. train.csv.
            
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