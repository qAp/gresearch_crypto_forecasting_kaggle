
import numpy as np  
import pandas as pd


class SimEnv:
    def __init__(self, csv_path='train.csv', num_sample=None):
        self.df = pd.read_csv(csv_path)

        timestamps = self.df['timestamp'].unique()

        if num_sample is None:
            val_split, test_split = .2, .15
            num_sample = int((val_split + test_split) * len(timestamps))

        self.num_sample = num_sample
        self.timestamps = timestamps[-num_sample:]

        self.reset()

    def reset(self):
        self._row_count = 0
        self.gt_df_list = []
        self.pr_df_list = []

    def iter_test(self):
        for i, timestamp in enumerate(self.timestamps):

            test_df = self.df[self.df['timestamp'] == timestamp].copy()

            row_id = self._row_count + np.arange(len(test_df))
            test_df['row_id'] = row_id

            gt_df = test_df[['timestamp', 'Asset_ID', 'Target', 'row_id']].copy()
            self.gt_df_list.append(gt_df)

            test_df.drop('Target', axis=1, inplace=True)

            sample_prediction_df = pd.DataFrame({'row_id': row_id})
            sample_prediction_df['Target'] = 0

            self._row_count += len(test_df)

            yield test_df, sample_prediction_df

    def predict(self, sample_prediction_df):
        self.pr_df_list.append(sample_prediction_df)

    def __len__(self):
        return self.num_sample
