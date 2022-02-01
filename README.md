
# G-Research Crypto Forecasting Kaggle
https://www.kaggle.com/c/g-research-crypto-forecasting/overview

Solution utilises the Spacetimeformer: https://github.com/QData/spacetimeformer, based on [Long-Range Transformers for Dynamic Spatiotemporal Forecasting](https://arxiv.org/abs/2109.12218) by Jake Grigsby, Zhe Wang, and Yanjun Qi.


## Data preprocessing

```bash
python spacetimeformer/data/crypto/preprocess.py
```

## Training

```bash
python spacetimeformer/train.py \
--gpus 0 \
--dset 'crypto' \
--xtra_target_cols {xtra_target_cols} \
--data_path '/kaggle/input/stfa01-data-preprocessing/train_tindex.feather' \
--batch_size 10 \
--context_points 128 \
--target_points 16 \
--start_token_len 8 \
--wandb \
--plot \
--attn_plot \
--max_epochs 3 \
--val_check_interval 0.1 \
--save_every_n_val_epochs 1 \
--load_from_checkpoint 'best.pth' \
--base_lr 1e-4
```

## Submission

```bash
cd /kaggle/working

python /kaggle/gresearch_crypto_forecasting_kaggle/spacetimeformer/crypto_submission.py \
--dset 'crypto' \
--xtra_target_cols {xtra_target_cols} \
--load_from_checkpoint {load_from_checkpoint} \
--data_path '/kaggle/input/stfa01-data-preprocessing/train_tindex.feather' \
--context_points 250 \
--target_points 16 \
--time_resolution 1 \
--start_token_len 8
```


