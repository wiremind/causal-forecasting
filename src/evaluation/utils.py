import json
import os
from glob import glob
from logging import Logger
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from src.data.mimic_iii.real_dataset import MIMIC3RealDatasetCollection
from src.rdd.utils import from_fully_qualified_import


def load_ct_model(
    seed: int, device: torch.device, dataset_config: dict, model_path: str
):
    args = DictConfig(
        {
            "model": {
                "dim_treatments": "???",
                "dim_vitals": "???",
                "dim_static_features": "???",
                "dim_outcomes": "???",
                "name": "CT",
                "multi": {
                    "_target_": "src.models.ct.CT",
                    "max_seq_length": "65",
                    "seq_hidden_units": 24,
                    "br_size": 22,
                    "fc_hidden_units": 22,
                    "dropout_rate": 0.2,
                    "num_layer": 2,
                    "num_heads": 3,
                    "max_grad_norm": None,
                    "batch_size": 64,
                    "attn_dropout": True,
                    "disable_cross_attention": False,
                    "isolate_subnetwork": "_",
                    "self_positional_encoding": {
                        "absolute": False,
                        "trainable": True,
                        "max_relative_position": 30,
                    },
                    "optimizer": {
                        "optimizer_cls": "adam",
                        "learning_rate": 0.0001,
                        "weight_decay": 0.0,
                        "lr_scheduler": False,
                    },
                    "augment_with_masked_vitals": True,
                    "tune_hparams": False,
                    "tune_range": 50,
                    "hparams_grid": None,
                    "resources_per_trial": None,
                },
            },
            "dataset": {
                "val_batch_size": 512,
                "treatment_mode": "multilabel",
                "_target_": "src.data.MIMIC3RealDatasetCollection",
                "seed": "${exp.seed}",
                "name": "mimic3_real",
                "path": "data/processed/all_hourly_data.h5",
                "min_seq_length": 30,
                "max_seq_length": 60,
                "max_number": 5000,
                "projection_horizon": 5,
                "split": {"val": 0.15, "test": 0.15},
                "autoregressive": True,
                "treatment_list": ["vaso", "vent"],
                "outcome_list": ["diastolic blood pressure"],
                "vital_list": [
                    "heart rate",
                    "red blood cell count",
                    "sodium",
                    "mean blood pressure",
                    "systemic vascular resistance",
                    "glucose",
                    "chloride urine",
                    "glascow coma scale total",
                    "hematocrit",
                    "positive end-expiratory pressure set",
                    "respiratory rate",
                    "prothrombin time pt",
                    "cholesterol",
                    "hemoglobin",
                    "creatinine",
                    "blood urea nitrogen",
                    "bicarbonate",
                    "calcium ionized",
                    "partial pressure of carbon dioxide",
                    "magnesium",
                    "anion gap",
                    "phosphorous",
                    "venous pvo2",
                    "platelets",
                    "calcium urine",
                ],
                "static_list": ["gender", "ethnicity", "age"],
                "drop_first": False,
            },
            "exp": {
                "seed": 10,
                "gpus": [0],
                "max_epochs": 1,
                "logging": False,
                "mlflow_uri": "http://127.0.0.1:5000",
                "unscale_rmse": True,
                "percentage_rmse": False,
                "alpha": 0.01,
                "update_alpha": True,
                "alpha_rate": "exp",
                "balancing": "domain_confusion",
                "bce_weight": False,
                "weights_ema": True,
                "beta": 0.99,
            },
        }
    )

    dataset_collection = MIMIC3RealDatasetCollection(
        dataset_config["path"],
        min_seq_length=dataset_config["min_seq_length"],
        max_seq_length=dataset_config["max_seq_length"],
        seed=seed,
        max_number=dataset_config["max_number"],
        split=dataset_config["split"],
        projection_horizon=dataset_config["projection_horizon"],
        autoregressive=dataset_config["autoregressive"],
        outcome_list=dataset_config["outcome_list"],
        vitals=dataset_config["vital_list"],
        treatment_list=dataset_config["treatment_list"],
        static_list=dataset_config["static_list"],
    )
    dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data["outputs"].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data[
        "current_treatments"
    ].shape[-1]
    args.model.dim_vitals = (
        dataset_collection.train_f.data["vitals"].shape[-1]
        if dataset_collection.has_vitals
        else 0
    )
    args.model.dim_static_features = dataset_collection.train_f.data[
        "static_features"
    ].shape[-1]
    multimodel = instantiate(
        args.model.multi, args, dataset_collection, _recursive_=False
    )
    multimodel.hparams.exp.weights_ema = False
    multimodel.load_state_dict(
        torch.load(
            os.path.join(
                model_path,
                "checkpoints",
                os.listdir(os.path.join(model_path, "checkpoints"))[0],
            ),
            map_location=device,
        )["state_dict"]
    )

    multimodel_trainer = Trainer(
        devices=args.exp.gpus,
        max_epochs=args.exp.max_epochs,
        gradient_clip_val=args.model.multi.max_grad_norm,
    )
    multimodel.trainer = multimodel_trainer
    multimodel = multimodel.double()
    multimodel = multimodel.eval()
    multimodel = multimodel.to(device)

    return multimodel


def format_models_dict(
    dataset_config: dict[str, Any],
    device: torch.device,
    seeds: list[int],
    tft_models_definition: dict | None = None,
    ct_models_path: str | None = None,
) -> dict[int, dict[str, tuple[Any, str] | tuple[Any, None]]]:
    models_dict_per_seed = {}
    if tft_models_definition is not None:
        for model_name_prefix, folder_name_prefix, model_class in tft_models_definition:
            for seed_idx, seed in enumerate(seeds):
                models_dict_per_seed.setdefault(seed, {})[
                    f"{model_name_prefix}_{seed_idx}"
                ] = (
                    model_class,
                    glob(
                        os.path.join(
                            f"{folder_name_prefix}_{seed_idx}", "checkpoints/*.ckpt"
                        )
                    )[-1],
                )
    if ct_models_path is not None:
        for seed_index, seed in enumerate(seeds):
            models_dict_per_seed.setdefault(seed, {})[f"CT_{seed_index}"] = (
                lambda _seed, _seed_idx: load_ct_model(
                    device=device,
                    seed=_seed,
                    dataset_config=dataset_config,
                    model_path=os.path.join(ct_models_path, str(_seed_idx)),
                ),
                None,
            )
    return models_dict_per_seed


def load_evaluation_model(
    model_class: type,
    model_name: str,
    seed: int,
    seed_idx: int,
    time_shift: int,
    model_path: str,
    device: torch.device,
):
    is_ct_model = model_name.startswith("CT")
    if isinstance(model_class, str):
        model_class = from_fully_qualified_import(model_class)
    if is_ct_model:
        model = model_class(_seed=seed, _seed_idx=seed_idx)
        model.hparams.dataset.projection_horizon = 1 + time_shift
    else:
        model = model_class.load_from_checkpoint(model_path, map_location=device)
        if hasattr(model, "using_theta"):
            model.using_theta = True
    model = model.eval()
    model.freeze()
    model = model.to(device)

    return model


def save_metrics(
    destination_file_path: str, metrics_dict: dict[str, Any], logger: Logger
):
    with open(destination_file_path, "w") as f:
        f.truncate(0)
        f.seek(0)
        json.dump(metrics_dict, f, indent=4)

    logger.info(json.dumps(metrics_dict, indent=4))
