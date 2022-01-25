

import gresearch_crypto
from spacetimeformer.train import create_parser
from spacetimeformer.crypto_predictor import CryptoPredictor



env = gresearch_crypto.make_env()
iter_test = env.iter_test()

parser = create_parser()
args = parser.parse_args()

predictor = CryptoPredictor(args)

predictor.initialize_context()

for test_df, sample_prediction_df in iter_test:

    raw_df = predictor._get_raw_df(test_df, sample_prediction_df)
    df = predictor._preprocess_features(raw_df)
    xc, yc, xt, yt = predictor.frame2tensors(df)
    pr = predictor.predict(xc, yc, xt, yt)
    predictor.update_context(test_df, pr)

    sample_prediction_df['Target'] = pr[test_df['Asset_ID'].values]
    env.predict(sample_prediction_df)
