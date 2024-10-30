import logging

import numpy as np
import pandas as pd

from src.data.mimic_iii.real_dataset import MIMIC3RealDataset

logger = logging.getLogger(__name__)


class MIMIC3TFTRealDataset(MIMIC3RealDataset):
    """
    Pytorch-style real-world MIMIC-III dataset for TFT models
    """

    def __init__(
        self,
        treatments: pd.DataFrame,
        outcomes: pd.DataFrame,
        vitals: pd.DataFrame,
        static_features: pd.DataFrame,
        outcomes_unscaled: pd.DataFrame,
        scaling_params: dict,
        subset_name: str,
    ):
        """
        Args:
            treatments: DataFrame with treatments; multiindex by (patient_id, timestep)
            outcomes: DataFrame with outcomes; multiindex by (patient_id, timestep)
            vitals: DataFrame with vitals (time-varying covariates); multiindex by (patient_id, timestep)
            static_features: DataFrame with static features
            outcomes_unscaled: DataFrame with unscaled outcomes; multiindex by (patient_id, timestep)
            scaling_params: Standard normalization scaling parameters
            subset_name: train / val / test
        """
        user_sizes = vitals.groupby("subject_id").size()

        processed_treatments = (
            treatments.unstack(fill_value=np.nan, level=0)
            .stack(dropna=False)
            .swaplevel(0, 1)
            .sort_index()
        )
        processed_outcomes = (
            outcomes.unstack(fill_value=np.nan, level=0)
            .stack(dropna=False)
            .swaplevel(0, 1)
            .sort_index()
        )
        processed_outcomes_unscaled = (
            outcomes_unscaled.unstack(fill_value=np.nan, level=0)
            .stack(dropna=False)
            .swaplevel(0, 1)
            .sort_index()
        )
        processed_vitals = (
            vitals.unstack(fill_value=np.nan, level=0)
            .stack(dropna=False)
            .swaplevel(0, 1)
            .sort_index()
        )

        active_entries = (~processed_treatments.isna().any(axis=1)).astype(float)
        active_entries = active_entries.values.reshape(
            (len(user_sizes), max(user_sizes), 1)
        )

        processed_treatments = (
            processed_treatments.fillna(0.0)
            .values.reshape((len(user_sizes), max(user_sizes), -1))
            .astype(float)
        )
        processed_outcomes = processed_outcomes.fillna(0.0).values.reshape(
            (len(user_sizes), max(user_sizes), -1)
        )
        processed_vitals = processed_vitals.fillna(0.0).values.reshape(
            (len(user_sizes), max(user_sizes), -1)
        )
        processed_outcomes_unscaled = processed_outcomes_unscaled.fillna(
            0.0
        ).values.reshape((len(user_sizes), max(user_sizes), -1))

        super().__init__(
            treatments,
            outcomes,
            vitals,
            static_features,
            outcomes_unscaled,
            scaling_params,
            subset_name,
        )

        # TFT dataset
        self.data.update(
            {
                "vitals": processed_vitals,
                "current_treatments": processed_treatments,
                "active_entries": active_entries,
                "outputs": processed_outcomes,
                "unscaled_outputs": processed_outcomes_unscaled,
            }
        )

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info(f"Shape of TFT processed {self.subset_name} data: {data_shapes}")
