import logging
from typing import Any

import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.evaluation.utils import format_models_dict, save_metrics
from src.rdd.rdd_rmse import compute_rdd_metrics_for_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_weighted_metric_values(
    rmse_metric_values: np.ndarray,
    mae_metric_values: np.ndarray,
    metrics_weights: list[float],
):
    rmse_weighted_average = np.average(rmse_metric_values, weights=metrics_weights)
    mae_weighted_average = np.average(mae_metric_values, weights=metrics_weights)
    return {
        "rmse": {
            "values": rmse_metric_values,
            "mean": np.mean(rmse_metric_values),
            "std": np.std(rmse_metric_values),
            "weighted_mean": rmse_weighted_average,
            "weighted_std": np.round(
                np.sqrt(
                    np.average(
                        (np.array(rmse_metric_values) - rmse_weighted_average) ** 2,
                        weights=metrics_weights,
                    )
                ),
                8,
            ),
        },
        "mae": {
            "values": mae_metric_values,
            "mean": np.mean(mae_metric_values),
            "std": np.std(mae_metric_values),
            "weighted_mean": mae_weighted_average,
            "weighted_std": np.round(
                np.sqrt(
                    np.average(
                        (np.array(mae_metric_values) - mae_weighted_average) ** 2,
                        weights=metrics_weights,
                    )
                ),
                8,
            ),
        },
    }


def compute_metrics_from_values(
    rdd_df_per_time_per_seed: dict[int, dict[int, pd.DataFrame]],
    seeds: list[int],
    rdd_path: str,
    rdd_query_str: str,
    top_percent_outliers_selection: float,
    models_dict_per_seed: dict[int, dict[str, tuple[Any, str] | tuple[Any, None]]],
    tft_models_definition: list[tuple[str, str, Any]],
    compute_ct_values: bool,
):
    # initalize result dict
    metrics_dict = {
        "TIME_SHIFTS": list(rdd_df_per_time_per_seed.keys()),
        "RDD_PATH": rdd_path,
        "rdd_dataset_len_per_time_per_seed": {
            time_shift: {
                seed: len(rdd_df_per_time_per_seed[time_shift][seed]) for seed in seeds
            }
            for time_shift in rdd_df_per_time_per_seed.keys()
        },
        "rdd_dataset_filtering": {
            "query_str": rdd_query_str,
            "top_percent_outliers_filtereing": top_percent_outliers_selection,
            "total_filtering_percent_per_time_shift": {
                time_shift: {
                    seed: (
                        rdd_df_per_time_per_seed[time_shift][seed]
                        .eval(rdd_query_str)
                        .mean()
                        - (top_percent_outliers_selection * 2)
                    )
                    for seed in seeds
                }
                for time_shift in rdd_df_per_time_per_seed.keys()
            },
        },
        "models_per_seed": {
            seed: {
                model_name: model_path
                for model_name, (_, model_path) in models_dict.items()
                if "CT_" not in model_name
            }
            for seed, models_dict in models_dict_per_seed.items()
        },
        "metrics_per_time_shift": {
            time_shift: {
                "raw_metrics": {
                    "metrics_per_seed": {},
                    "metrics_per_architecture": {},
                },
                "filtered_metrics": {
                    "metrics_per_seed": {},
                    "metrics_per_architecture": {},
                },
            }
            for time_shift in rdd_df_per_time_per_seed.keys()
        },
    }

    # compute metrics
    for time_shift in rdd_df_per_time_per_seed.keys():
        for seed, models_dict in models_dict_per_seed.items():
            seed_df = rdd_df_per_time_per_seed[time_shift][seed]
            for model_name in models_dict.keys():
                # Raw metrics
                metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"][
                    "metrics_per_seed"
                ].setdefault(seed, {}).setdefault("rmse", {})[
                    model_name
                ] = root_mean_squared_error(
                    y_true=seed_df["rdd_delta"],
                    y_pred=seed_df[model_name + "_delta_demand"],
                )
                metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"][
                    "metrics_per_seed"
                ][seed].setdefault("mae", {})[model_name] = mean_absolute_error(
                    y_true=seed_df["rdd_delta"],
                    y_pred=seed_df[model_name + "_delta_demand"],
                )
                if rdd_query_str:
                    # Filtered metrics
                    filtered_df = seed_df.query(rdd_query_str)
                    if top_percent_outliers_selection:
                        q1, q2 = (
                            filtered_df["rdd_delta"]
                            .quantile(
                                q=[
                                    top_percent_outliers_selection,
                                    1 - top_percent_outliers_selection,
                                ]
                            )
                            .values.tolist()
                        )
                        filtered_df.query("@q1 <= rdd_delta <= @q2", inplace=True)
                    metrics_dict["metrics_per_time_shift"][time_shift][
                        "filtered_metrics"
                    ]["metrics_per_seed"].setdefault(seed, {}).setdefault("rmse", {})[
                        model_name
                    ] = root_mean_squared_error(
                        y_true=filtered_df["rdd_delta"],
                        y_pred=filtered_df[model_name + "_delta_demand"],
                    )
                    metrics_dict["metrics_per_time_shift"][time_shift][
                        "filtered_metrics"
                    ]["metrics_per_seed"][seed].setdefault("mae", {})[
                        model_name
                    ] = mean_absolute_error(
                        y_true=filtered_df["rdd_delta"],
                        y_pred=filtered_df[model_name + "_delta_demand"],
                    )

        # format metrics per architecture
        metrics_weights = list(
            metrics_dict["rdd_dataset_len_per_time_per_seed"][time_shift].values()
        )
        metrics_filtered_weights = [
            metrics_dict["rdd_dataset_filtering"][
                "total_filtering_percent_per_time_shift"
            ][time_shift][seeds[idx]]
            * rdd_dataset_len
            for idx, rdd_dataset_len in enumerate(metrics_weights)
        ]

        models_prefix = [model_prefix for model_prefix, *_ in tft_models_definition]
        if compute_ct_values:
            models_prefix.append("CT")

        for model_prefix in models_prefix:
            metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"][
                "metrics_per_architecture"
            ][model_prefix] = format_weighted_metric_values(
                rmse_metric_values=np.round(
                    [
                        metrics_dict["metrics_per_time_shift"][time_shift][
                            "raw_metrics"
                        ]["metrics_per_seed"][seed]["rmse"][f"{model_prefix}_{idx}"]
                        for idx, seed in enumerate(seeds)
                    ],
                    8,
                ).tolist(),
                mae_metric_values=np.round(
                    [
                        metrics_dict["metrics_per_time_shift"][time_shift][
                            "raw_metrics"
                        ]["metrics_per_seed"][seed]["mae"][f"{model_prefix}_{idx}"]
                        for idx, seed in enumerate(seeds)
                    ],
                    8,
                ).tolist(),
                metrics_weights=metrics_weights,
            )
            if rdd_query_str:
                metrics_dict["metrics_per_time_shift"][time_shift]["filtered_metrics"][
                    "metrics_per_architecture"
                ][model_prefix] = format_weighted_metric_values(
                    rmse_metric_values=[
                        metrics_dict["metrics_per_time_shift"][time_shift][
                            "filtered_metrics"
                        ]["metrics_per_seed"][seed]["rmse"][f"{model_prefix}_{idx}"]
                        for idx, seed in enumerate(seeds)
                    ],
                    mae_metric_values=[
                        metrics_dict["metrics_per_time_shift"][time_shift][
                            "filtered_metrics"
                        ]["metrics_per_seed"][seed]["mae"][f"{model_prefix}_{idx}"]
                        for idx, seed in enumerate(seeds)
                    ],
                    metrics_weights=metrics_filtered_weights,
                )

    # format values as the one displayed in the paper
    metrics_dict["paper_metrics_per_time_shift"] = {
        time_shift: {
            model_name: {
                "average_rmse": np.round(metrics_values["rmse"]["weighted_mean"], 3),
                "std_rmse": np.round(metrics_values["rmse"]["weighted_std"], 3),
                "average_mae": np.round(metrics_values["mae"]["weighted_mean"], 3),
                "std_mae": np.round(metrics_values["mae"]["weighted_std"], 3),
            }
            for model_name, metrics_values in metrics_dict["metrics_per_time_shift"][
                time_shift
            ]["filtered_metrics"]["metrics_per_architecture"].items()
        }
        for time_shift in rdd_df_per_time_per_seed.keys()
    }

    return metrics_dict


@hydra.main(config_name="config.yaml", config_path="./config/", version_base="1.3.2")
def main(args: DictConfig):

    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    device = torch.device("cuda")
    seeds = list(args.metrics.seeds)
    n_jobs = len(seeds)

    # Fetch checkpoint paths
    models_dict_per_seed = format_models_dict(
        dataset_config=dict(args.dataset),
        device=device,
        seeds=seeds,
        tft_models_definition=list(args.metrics.get("tft_models") or []),
        ct_models_path=args.metrics.ct_models_path,
    )

    # Reading RDD dataset
    rdd_dataset = pd.read_parquet(args.rdd.destination_file_path).astype(
        {"subject_id": int, "hours_in": int}
    )
    rdd_df_per_time_per_seed: dict[int, dict[int, pd.DataFrame]] = {
        time_shift: {} for time_shift in range(args.metrics.max_projection_step)
    }
    for time_shift in rdd_df_per_time_per_seed.keys():
        # Forecast values
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(compute_rdd_metrics_for_seed)(
                rdd_dataset_df=rdd_dataset,
                seed=seed,
                seed_idx=seeds.index(seed),
                models_dict=models_dict,
                device=device,
                dataset_config=dict(args.dataset),
                time_shift=time_shift,
            )
            for seed, models_dict in models_dict_per_seed.items()
        )

        # Storing forecasted values
        for seed, seed_rdd_df in results:
            rdd_df_per_time_per_seed[time_shift][seed] = seed_rdd_df

    # Compute metrics per seed
    metrics_dict = compute_metrics_from_values(
        rdd_df_per_time_per_seed=rdd_df_per_time_per_seed,
        seeds=seeds,
        rdd_path=args.metrics.rdd.destination_file_path,
        rdd_query_str=args.metrics.rdd.rdd_query_str,
        top_percent_outliers_selection=args.metrics.rdd.top_percent_outliers_selection,
        models_dict_per_seed=models_dict_per_seed,
        tft_models_definition=list(args.metrics.get("tft_models") or []),
        compute_ct_values=bool(args.metrics.ct_models_path),
    )

    # compute metrics per with filtered
    save_metrics(
        destination_file_path=args.metrics.rdd.destination_file_path,
        metrics_dict=metrics_dict,
        logger=logger,
    )


if __name__ == "__main__":
    main()
