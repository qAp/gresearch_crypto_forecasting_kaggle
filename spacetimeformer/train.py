from argparse import ArgumentParser
import random
import os, sys
import warnings

import pytorch_lightning as pl
from spacetimeformer.data.crypto.config import DIR_PREPROCESS
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

import spacetimeformer as stf

_MODELS = ["spacetimeformer", "mtgnn", "lstm", "lstnet", "linear"]

_DSETS = [
    "asos",
    "metr-la",
    "pems-bay",
    "exchange",
    "precip",
    "toy1",
    "toy2",
    "solar_energy",
]


def create_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--model", type=str, default='spacetimeformer')
    parser.add_argument("--dset", type=str, default='asos')

    args, _ = parser.parse_known_args()

    if args.dset == "precip":
        stf.data.precip.GeoDset.add_cli(parser)
        stf.data.precip.CONUS_Precip.add_cli(parser)
        stf.data.DataModule.add_cli(parser)
    elif args.dset == "metr-la" or args.dset == "pems-bay":
        stf.data.metr_la.METR_LA_Data.add_cli(parser)
        stf.data.DataModule.add_cli(parser)
    else:
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
        stf.data.DataModule.add_cli(parser)

    if args.model == "lstm":
        stf.lstm_model.LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif args.model == "lstnet":
        stf.lstnet_model.LSTNet_Forecaster.add_cli(parser)
    elif args.model == "mtgnn":
        stf.mtgnn_model.MTGNN_Forecaster.add_cli(parser)
    elif args.model == "spacetimeformer":
        stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli(parser)
    elif args.model == "linear":
        stf.linear_model.Linear_Forecaster.add_cli(parser)

    stf.callbacks.TimeMaskedLossCallback.add_cli(parser)

    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--attn_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    parser.add_argument('--help', '-h', action='help')
    return parser


def create_model(config, x_dim=None, y_dim=None):
    if config.dset == "metr-la":
        x_dim = 2
        y_dim = 207
    elif config.dset == "pems-bay":
        x_dim = 2
        y_dim = 325
    elif config.dset == "precip":
        x_dim = 2
        y_dim = 49
    elif config.dset == "asos":
        x_dim = 6
        y_dim = 6
    elif config.dset == "solar_energy":
        x_dim = 6
        y_dim = 137
    elif config.dset == "exchange":
        x_dim = 6
        y_dim = 8
    elif config.dset == "toy1":
        x_dim = 6
        y_dim = 20
    elif config.dset == "toy2":
        x_dim = 6
        y_dim = 20

    assert x_dim is not None
    assert y_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_y=y_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "mtgnn":
        forecaster = stf.mtgnn_model.MTGNN_Forecaster(
            d_y=y_dim,
            d_x=x_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            gcn_depth=config.gcn_depth,
            dropout_p=config.dropout_p,
            node_dim=config.node_dim,
            dilation_exponential=config.dilation_exponential,
            conv_channels=config.conv_channels,
            subgraph_size=config.subgraph_size,
            skip_channels=config.skip_channels,
            end_channels=config.end_channels,
            residual_channels=config.residual_channels,
            layers=config.layers,
            propalpha=config.propalpha,
            tanhalpha=config.tanhalpha,
            learning_rate=config.learning_rate,
            kernel_size=config.kernel_size,
            l2_coeff=config.l2_coeff,
            time_emb_dim=config.time_emb_dim,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "lstnet":
        forecaster = stf.lstnet_model.LSTNet_Forecaster(
            context_points=config.context_points,
            d_y=y_dim,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            hidSkip=config.hidSkip,
            CNN_kernel=config.CNN_kernel,
            skip=config.skip,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
        )
    elif config.model == "spacetimeformer":
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_y=y_dim,
            d_x=x_dim,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_token=config.dropout_token,
            dropout_attn_out=config.dropout_attn_out,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            post_norm=config.post_norm,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            class_loss_imp=config.class_loss_imp,
            time_emb_dim=config.time_emb_dim,
        )
    elif config.model == "linear":
        forecaster = stf.linear_model.Linear_Forecaster(
            context_points=config.context_points,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
        )

    return forecaster


def create_dset(config):
    INV_SCALER = lambda x: x
    NULL_VAL = None
    x_dim = None
    y_dim = None

    if config.dset == "metr-la" or config.dset == "pems-bay":
        if config.dset == "pems-bay":
            assert (
                "pems_bay" in config.data_path
            ), "Make sure to switch to the pems-bay file!"
        data = stf.data.metr_la.METR_LA_Data(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metr_la.METR_LA_Torch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
        )
        INV_SCALER = data.inverse_scale
        NULL_VAL = 0.0

    elif config.dset == "precip":
        dset = stf.data.precip.GeoDset(dset_dir=config.dset_dir, var="precip")
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.precip.CONUS_Precip,
            dataset_kwargs={
                "dset": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
            },
            batch_size=config.batch_size,
            workers=config.workers,
        )
        NULL_VAL = -1.0

    elif config.dset == 'crypto':
        from spacetimeformer.data.crypto.config import (DIR_PREPROCESS, 
                                                        ASSET_IDS, FEATURES,
                                                        TIME_FEATURES)
        from spacetimeformer.data.crypto import CryptoTimeSeries, CryptoDataset

        if config.data_path == 'auto':
            data_path = f'{DIR_PREPROCESS}/train_tindex.feather'
        else:
            data_path = config.data_path

        target_cols = [f'Target_{id}' for id in ASSET_IDS]
        feature_cols = ['Close_10']
            # f'{feature}_{id}' for id in ASSET_IDS 
            # for feature in FEATURES if feature in ('Close', 'VWAP')]

        val_split = .2
        test_split = .15
        NULL_VAL = -999

        dset = CryptoTimeSeries(data_path, target_cols, feature_cols,
                                val_split, test_split, NULL_VAL)

        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution},
            batch_size=config.batch_size,
            workers=config.workers)

        INV_SCALER = dset.reverse_scaling
        x_dim = len(TIME_FEATURES) + len(feature_cols)
        y_dim = len(target_cols)

    else:
        data_path = config.data_path
        if config.dset == "asos":
            if data_path == "auto":
                data_path = "./data/temperature-v1.csv"
            target_cols = ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"]

        elif config.dset == "solar_energy":
            if data_path == "auto":
                data_path = "./data/solar_AL_converted.csv"
            target_cols = [str(i) for i in range(137)]

        elif "toy" in config.dset:
            if data_path == "auto":
                if config.dset == "toy1":
                    data_path = "./data/toy_dset1.csv"
                elif config.dset == "toy2":
                    data_path = "./data/toy_dset2.csv"
                else:
                    raise ValueError(f"Unrecognized toy dataset {config.dset}")
            target_cols = [f"D{i}" for i in range(1, 21)]

        elif config.dset == "exchange":
            if data_path == "auto":
                data_path = "./data/exchange_rate_converted.csv"
            target_cols = [
                "Australia",
                "United Kingdom",
                "Canada",
                "Switzerland",
                "China",
                "Japan",
                "New Zealand",
                "Singapore",
            ]

        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols)

        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
        )
        INV_SCALER = dset.reverse_scaling
        NULL_VAL = None

    return DATA_MODULE, INV_SCALER, NULL_VAL, x_dim, y_dim


def create_callbacks(config):
    saving = pl.callbacks.ModelCheckpoint(
        dirpath="/kaggle/working/stf_model_checkpoints",
        monitor="val/mse",
        auto_insert_metric_name=False,
        filename="epoch{epoch:02d}-val_loss{val/loss:.3f}-val_mse{val/mse:.3f}",
        save_top_k=1,
    )
    callbacks = [saving]

    if config.early_stopping:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss",
                patience=5,
            )
        )
    if config.wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                epochs=config.teacher_forcing_anneal_epochs,
            )
        )
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.target_points,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks


def main(args):
    if args.wandb:
        """
        Set Up Weights and Biases online logging
        by entering your org and project names
        in the variables below.
        """
        project = 'gresearch'  # wandb project name
        wandb_dir = '/kaggle/working/stf_LOG_DIR'


        assert (
            project is not None
        ), "Please edit train.py with your wandb account information."

        os.makedirs(wandb_dir, exist_ok=True)

        experiment = wandb.init(
            project=project,
            config=args,
            dir=wandb_dir,
            reinit=True,
        )
        config = wandb.config
        wandb.run.save()
    else:
        config = args

    # Dset
    data_module, inv_scaler, null_val, x_dim, y_dim = create_dset(config)

    # Model
    forecaster = create_model(config, x_dim, y_dim)    
    forecaster.set_inv_scaler(inv_scaler)

    # Callbacks
    callbacks = create_callbacks(config)
    test_samples = next(iter(data_module.test_dataloader()))

    if config.wandb and config.plot:
        callbacks.append(
            stf.plot.PredictionPlotterCallback(
                test_samples, total_samples=min(8, config.batch_size)
            )
        )
    if config.wandb and config.model == "spacetimeformer" and config.attn_plot:

        callbacks.append(
            stf.plot.AttentionMatrixCallback(
                test_samples,
                layer=0,
                total_samples=min(16, config.batch_size),
                raw_data_dir=wandb.run.dir,
            )
        )

    # Deal with missing entries in some datasets
    if null_val is not None:
        forecaster.set_null_value(null_val)

    # Logging
    if config.wandb:
        logger = pl.loggers.WandbLogger(experiment=experiment, 
                                        save_dir=wandb_dir)
        logger.log_hyperparams(config)

    trainer = pl.Trainer(
        gpus=config.gpus,
        callbacks=callbacks,
        logger=logger if args.wandb else None,
        accelerator="dp",
        log_gpu_memory=True,
        gradient_clip_val=config.grad_clip_norm,
        gradient_clip_algorithm="norm",
        overfit_batches=20 if config.debug else 0,
        # track_grad_norm=2,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        val_check_interval=0.25 if args.dset == "asos" else 1.0,
    )

    # Train
    trainer.fit(forecaster, datamodule=data_module)

    # Test
    trainer.test(datamodule=data_module, ckpt_path="best")

    if args.wandb:
        experiment.finish()


if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    if args.wandb:
        import wandb

    for trial in range(args.trials):
        main(args)
