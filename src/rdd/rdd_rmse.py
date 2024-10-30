from typing import Any

import numpy as np
import pandas as pd
from torch import nn, no_grad
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.mimic_iii.rdd_dataset import MIMIC3RDDRealDatasetCollection
from src.evaluation.utils import load_evaluation_model


def forecast_ate_values(
    test_dl: DataLoader,
    model: nn.Module,
    s_mean: float,
    s_std: float,
    time_shift: int,
    model_name: str,
):
    left_values = []
    right_values = []
    subject_ids = []
    hours_in = []
    device = model.device
    with no_grad():
        for batch in tqdm(
            test_dl, desc=f"Forecasting ATE values using model {model_name}"
        ):
            tau_values = batch["future_past_split"].numpy()
            batch_subject_ids = batch["subject_ids"].numpy()

            assert batch_subject_ids[0] == batch_subject_ids[1]
            assert tau_values[0] == tau_values[1]

            tau = int(tau_values[0])
            prediction_index = tau + time_shift
            subject_id = batch_subject_ids[0]

            for key in [
                "vitals",
                "static_features",
                "current_treatments",
                "outputs",
                "prev_treatments",
                "prev_outputs",
                "active_entries",
            ]:
                batch[key] = batch[key].to(device)

            output = model.forecast(batch)
            right_values.append(output[0, prediction_index, 0].item())
            left_values.append(output[1, prediction_index, 0].item())
            subject_ids.append(subject_id)
            hours_in.append(prediction_index)
            right_price, left_price = (
                batch["current_treatments"][0, prediction_index, :].cpu().numpy(),
                batch["current_treatments"][1, prediction_index, :].cpu().numpy(),
            )
            left_price = 2 * left_price[0] + left_price[1]  # 2*vaso + vent
            right_price = 2 * right_price[0] + right_price[1]

    left_values = np.array(left_values) * s_std + s_mean
    right_values = np.array(right_values) * s_std + s_mean
    ate_values = right_values - left_values

    return pd.DataFrame(
        {
            "subject_id": subject_ids,
            "hours_in": hours_in,
            "left_prediction": left_values,
            "right_prediction": right_values,
            "ate": ate_values,
        }
    ).astype({"subject_id": int, "hours_in": int})


def compute_rdd_metrics_for_seed(
    rdd_dataset_df: pd.DataFrame,
    seed: int,
    seed_idx: int,
    dataset_config: dict[str, tuple[Any, Any]],
    models_dict: dict[str, Any],
    time_shift: int,
    device,
):

    print(f"Loading test dataset for seed {seed}")
    dataset_collection = MIMIC3RDDRealDatasetCollection(
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
    dataset_collection.test_f.preprocess_rdd_test_data(time_shift=time_shift)

    test_dataset_subject_ids = set(
        dataset_collection.test_f.data["subject_ids"].tolist()
    )
    seed_rdd_df = rdd_dataset_df.query("subject_id in @test_dataset_subject_ids")
    s_mean = dataset_collection.test_f.scaling_params["output_means"]
    s_std = dataset_collection.test_f.scaling_params["output_stds"]

    for model_name, (model_class, model_path) in tqdm(
        models_dict.items(), desc=f"predictions for seed {seed} time shift {time_shift}"
    ):
        # loading model
        model = load_evaluation_model(
            model_class=model_class,
            model_name=model_name,
            seed=seed,
            seed_idx=seed_idx,
            time_shift=time_shift,
            model_path=model_path,
            device=device,
        )

        # creating dataloader. Always use a batchsize of 2
        test_dl = DataLoader(dataset_collection.test_f, batch_size=2, shuffle=False)

        # predicting values
        ate_df = forecast_ate_values(
            test_dl=test_dl,
            model=model,
            s_mean=s_mean,
            s_std=s_std,
            time_shift=time_shift,
            model_name=model_name,
        )

        seed_rdd_df = seed_rdd_df.merge(
            ate_df[
                ["subject_id", "hours_in", "left_prediction", "right_prediction"]
            ].rename(
                columns={
                    "left_prediction": model_name + "_left_predicted_demand",
                    "right_prediction": model_name + "_right_predicted_demand",
                }
            ),
            on=["subject_id", "hours_in"],
            how="inner",
        )

        seed_rdd_df[model_name + "_delta_demand"] = (
            seed_rdd_df[model_name + "_right_predicted_demand"]
            - seed_rdd_df[model_name + "_left_predicted_demand"]
        )

    seed_rdd_df["rdd_delta"] = seed_rdd_df["CATE"]
    return seed, seed_rdd_df
