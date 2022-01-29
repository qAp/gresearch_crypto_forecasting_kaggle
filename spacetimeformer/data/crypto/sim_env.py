
import numpy as np
import pandas as pd


class SimEnv:
    def __init__(self, csv_path='train.csv', num_sample=10_000):
        self.df = pd.read_csv(csv_path)
        self.df.drop('Target', axis=1, inplace=True)
        self.num_sample = num_sample

        self.timestamps = self.df['timestamp'].unique()[:num_sample]

        self._row_count = 0

#     def iter_test(self):
#         for i, timestamp in enumerate(self.timestamps):

#             test_df = self.df[self.df['timestamp'] == timestamp]
#             test_df = pd.DataFrame({'Asset_ID': ASSET_IDS}).merge(test_df, how='outer')
#             test_df['timestamp'].fillna(timestamp, inplace=True)

#             row_id = i * len(ASSET_IDS) + np.arange(len(ASSET_IDS))
#             test_df['row_id'] = row_id

#             sample_prediction_df = pd.DataFrame({'row_id': row_id})
#             sample_prediction_df['Target'] = 0

#             yield test_df, sample_prediction_df

    def reset(self):
        self._row_count = 0

    def iter_test(self):
        for i, timestamp in enumerate(self.timestamps):

            test_df = self.df[self.df['timestamp'] == timestamp].copy()

            row_id = self._row_count + np.arange(len(test_df))
            test_df['row_id'] = row_id

            sample_prediction_df = pd.DataFrame({'row_id': row_id})
            sample_prediction_df['Target'] = 0

            self._row_count += len(test_df)

            yield test_df, sample_prediction_df

    def __len__(self):
        return self.num_sample
