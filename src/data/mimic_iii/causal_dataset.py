from copy import deepcopy

from sklearn.model_selection import train_test_split

from src import ROOT_PATH
from src.data.mimic_iii.load_data import load_mimic3_data_processed
from src.data.mimic_iii.real_dataset import RealDatasetCollection
from src.data.mimic_iii.tft_dataset import MIMIC3TFTRealDataset


class MIMIC3TFTDatasetCollectionCausal(RealDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_f) for training TFT causal models
    """

    def __init__(
        self,
        path: str,
        min_seq_length: int = 30,
        max_seq_length: int = 60,
        seed: int = 100,
        max_number: int = None,
        split: dict = {"val": 0.2, "test": 0.2},
        split_causal: dict = {"S1": 0.8},
        projection_horizon: int = 5,
        autoregressive=True,
        dataset_class=MIMIC3TFTRealDataset,
        **kwargs
    ):
        """
        Args:
            path: Path with MIMIC-3 dataset (HDFStore)
            min_seq_length: Min sequence lenght in cohort
            max_seq_length: Max sequence lenght in cohort
            seed: Seed for random cohort patient selection
            max_number: Maximum number of patients in cohort
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            autoregressive:
        """
        super().__init__()
        self.seed = seed
        (
            treatments,
            outcomes,
            vitals,
            static_features,
            outcomes_unscaled,
            scaling_params,
        ) = load_mimic3_data_processed(
            ROOT_PATH + "/" + path,
            min_seq_length=min_seq_length,
            max_seq_length=max_seq_length,
            max_number=max_number,
            data_seed=seed,
            **kwargs
        )

        # Train/test random_split
        static_features, static_features_test = train_test_split(
            static_features, test_size=split["test"], random_state=seed
        )
        (
            treatments,
            outcomes,
            vitals,
            outcomes_unscaled,
            treatments_test,
            outcomes_test,
            vitals_test,
            outcomes_unscaled_test,
        ) = (
            treatments.loc[static_features.index],
            outcomes.loc[static_features.index],
            vitals.loc[static_features.index],
            outcomes_unscaled.loc[static_features.index],
            treatments.loc[static_features_test.index],
            outcomes.loc[static_features_test.index],
            vitals.loc[static_features_test.index],
            outcomes_unscaled.loc[static_features_test.index],
        )

        # Train/val
        static_features_train, static_features_val = train_test_split(
            static_features,
            test_size=split["val"] / (1 - split["test"]),
            random_state=2 * seed,
        )
        (
            treatments_train,
            outcomes_train,
            vitals_train,
            outcomes_unscaled_train,
            treatments_val,
            outcomes_val,
            vitals_val,
            outcomes_unscaled_val,
        ) = (
            treatments.loc[static_features_train.index],
            outcomes.loc[static_features_train.index],
            vitals.loc[static_features_train.index],
            outcomes_unscaled.loc[static_features_train.index],
            treatments.loc[static_features_val.index],
            outcomes.loc[static_features_val.index],
            vitals.loc[static_features_val.index],
            outcomes_unscaled.loc[static_features_val.index],
        )

        # Train S1/S2
        static_features_train_s1, static_features_train_s2 = train_test_split(
            static_features_train, train_size=split_causal["S1"], random_state=seed
        )
        (
            treatments_train_s1,
            outcomes_train_s1,
            vitals_train_s1,
            outcomes_unscaled_train_s1,
            treatments_train_s2,
            outcomes_train_s2,
            vitals_train_s2,
            outcomes_unscaled_train_s2,
        ) = (
            treatments_train.loc[static_features_train_s1.index],
            outcomes_train.loc[static_features_train_s1.index],
            vitals_train.loc[static_features_train_s1.index],
            outcomes_unscaled_train.loc[static_features_train_s1.index],
            treatments_train.loc[static_features_train_s2.index],
            outcomes_train.loc[static_features_train_s2.index],
            vitals_train.loc[static_features_train_s2.index],
            outcomes_unscaled_train.loc[static_features_train_s2.index],
        )
        self.train_f_s1 = dataset_class(
            treatments_train_s1,
            outcomes_train_s1,
            vitals_train_s1,
            static_features_train_s1,
            outcomes_unscaled_train_s1,
            scaling_params,
            "train",
        )
        self.train_f_s2 = dataset_class(
            treatments_train_s2,
            outcomes_train_s2,
            vitals_train_s2,
            static_features_train_s2,
            outcomes_unscaled_train_s2,
            scaling_params,
            "train",
        )
        # Val S1/S2
        static_features_val_s1, static_features_val_s2 = train_test_split(
            static_features_val, train_size=split_causal["S1"], random_state=seed
        )
        (
            treatments_val_s1,
            outcomes_val_s1,
            vitals_val_s1,
            outcomes_unscaled_val_s1,
            treatments_val_s2,
            outcomes_val_s2,
            vitals_val_s2,
            outcomes_unscaled_val_s2,
        ) = (
            treatments_val.loc[static_features_val_s1.index],
            outcomes_val.loc[static_features_val_s1.index],
            vitals_val.loc[static_features_val_s1.index],
            outcomes_unscaled_val.loc[static_features_val_s1.index],
            treatments_val.loc[static_features_val_s2.index],
            outcomes_val.loc[static_features_val_s2.index],
            vitals_val.loc[static_features_val_s2.index],
            outcomes_unscaled_val.loc[static_features_val_s2.index],
        )
        self.val_f_s1 = dataset_class(
            treatments_val_s1,
            outcomes_val_s1,
            vitals_val_s1,
            static_features_val_s1,
            outcomes_unscaled_val_s1,
            scaling_params,
            "val",
        )
        self.val_f_s2 = dataset_class(
            treatments_val_s2,
            outcomes_val_s2,
            vitals_val_s2,
            static_features_val_s2,
            outcomes_unscaled_val_s2,
            scaling_params,
            "val",
        )

        self.test_f = dataset_class(
            treatments_test,
            outcomes_test,
            vitals_test,
            static_features_test,
            outcomes_unscaled_test,
            scaling_params,
            "test",
        )

        self.projection_horizon = projection_horizon
        self.has_vitals = True
        self.autoregressive = autoregressive
        self.processed_data_encoder = True

    def process_data_multi_val(self):
        """
        Used by CT
        """
        self.val_f_multi_s1 = deepcopy(self.val_f_s1)
        # Multiplying test trajectories
        self.val_f_multi_s1.explode_trajectories(self.projection_horizon)
        self.val_f_multi_s1.process_sequential_test(self.projection_horizon)
        self.val_f_multi_s1.process_sequential_multi(self.projection_horizon)

        self.val_f_multi_s2 = deepcopy(self.val_f_s2)
        # Multiplying test trajectories
        self.val_f_multi_s2.explode_trajectories(self.projection_horizon)
        self.val_f_multi_s2.process_sequential_test(self.projection_horizon)
        self.val_f_multi_s2.process_sequential_multi(self.projection_horizon)

    def process_data_multi_train(self):
        """
        Used by CT
        """
        self.train_f_multi_s1 = deepcopy(self.train_f_s1)
        # Multiplying test trajectories
        self.train_f_multi_s1.explode_trajectories(self.projection_horizon)
        self.train_f_multi_s1.process_sequential_test(self.projection_horizon)
        self.train_f_multi_s1.process_sequential_multi(self.projection_horizon)

        self.train_f_multi_s2 = deepcopy(self.train_f_s2)
        # Multiplying test trajectories
        self.train_f_multi_s2.explode_trajectories(self.projection_horizon)
        self.train_f_multi_s2.process_sequential_test(self.projection_horizon)
        self.train_f_multi_s2.process_sequential_multi(self.projection_horizon)
