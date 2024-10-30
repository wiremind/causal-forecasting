from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split

from src import ROOT_PATH
from src.data.mimic_iii.load_data import load_mimic3_data_processed
from src.data.mimic_iii.real_dataset import RealDatasetCollection
from src.data.mimic_iii.tft_dataset import MIMIC3TFTRealDataset
from src.rdd.rdd import (compute_subject_id_swicthing_time_steps_mapping,
                         rdd_indexes_iterator)


class MIMIC3RDDRealDataset(MIMIC3TFTRealDataset):
    def __init__(
        self,
        treatments,
        outcomes,
        vitals,
        static_features,
        outcomes_unscaled,
        scaling_params,
        subset_name,
    ):
        # keep raw treatments to use it for RDD switching time steps extraction
        self.raw_treatments = treatments.reset_index()
        super().__init__(
            treatments,
            outcomes,
            vitals,
            static_features,
            outcomes_unscaled,
            scaling_params,
            subset_name,
        )

    def preprocess_rdd_test_data(self, time_shift: int = 0):
        """Preprocess data to compute RDD rmse
        :param switching_time_steps_per_id: A dict mapping a subject id to the switching time steps
        """
        assert self.processed
        assert time_shift >= 0

        switching_time_steps_per_id = compute_subject_id_swicthing_time_steps_mapping(
            df=self.raw_treatments.copy()
        )

        outputs = self.data["outputs"]
        prev_outputs = self.data["prev_outputs"]
        sequence_lengths = self.data["sequence_lengths"]
        vitals = self.data["vitals"]
        next_vitals = self.data["next_vitals"]
        active_entries = self.data["active_entries"]
        current_treatments = self.data["current_treatments"]
        previous_treatments = self.data["prev_treatments"]
        static_features = self.data["static_features"]
        if "stabilized_weights" in self.data:
            stabilized_weights = self.data["stabilized_weights"]
        subject_ids = self.data["subject_ids"]

        num_patients, max_seq_length, num_features = outputs.shape
        num_seq2seq_rows = num_patients * max_seq_length

        seq2seq_previous_treatments = np.zeros(
            (num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1])
        )
        seq2seq_current_treatments = np.zeros(
            (num_seq2seq_rows, max_seq_length, current_treatments.shape[-1])
        )
        seq2seq_static_features = np.zeros(
            (num_seq2seq_rows, static_features.shape[-1])
        )
        seq2seq_outputs = np.zeros(
            (num_seq2seq_rows, max_seq_length, outputs.shape[-1])
        )
        seq2seq_prev_outputs = np.zeros(
            (num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1])
        )
        seq2seq_vitals = np.zeros((num_seq2seq_rows, max_seq_length, vitals.shape[-1]))
        seq2seq_next_vitals = np.zeros(
            (num_seq2seq_rows, max_seq_length - 1, next_vitals.shape[-1])
        )
        seq2seq_active_entries = np.zeros(
            (num_seq2seq_rows, max_seq_length, active_entries.shape[-1])
        )
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
        seq2seq_subject_ids = np.zeros(num_seq2seq_rows)
        if "stabilized_weights" in self.data:
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, max_seq_length))

        total_seq2seq_rows = 0  # we use this to shorten any trajectories later

        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])
            signal_id = subject_ids[i]
            for (*_, prediction_index,) in rdd_indexes_iterator(
                constant_treatments_indexes=switching_time_steps_per_id[signal_id],
            ):

                t = prediction_index
                if (t >= sequence_length) or (t - time_shift < 0):
                    continue

                # add reference sequence
                seq2seq_active_entries[
                    total_seq2seq_rows, : (t + 1), :
                ] = active_entries[i, : (t + 1), :]
                if "stabilized_weights" in self.data:
                    seq2seq_stabilized_weights[
                        total_seq2seq_rows, : (t + 1)
                    ] = stabilized_weights[i, : (t + 1)]
                seq2seq_previous_treatments[
                    total_seq2seq_rows, : (t + 1), :
                ] = previous_treatments[i, : (t + 1), :]
                seq2seq_current_treatments[
                    total_seq2seq_rows, : (t + 1), :
                ] = current_treatments[i, : (t + 1), :]
                seq2seq_outputs[total_seq2seq_rows, : (t + 1), :] = outputs[
                    i, : (t + 1), :
                ]
                seq2seq_prev_outputs[total_seq2seq_rows, : (t + 1), :] = prev_outputs[
                    i, : (t + 1), :
                ]
                seq2seq_vitals[total_seq2seq_rows, : (t + 1), :] = vitals[
                    i, : (t + 1), :
                ]
                seq2seq_next_vitals[
                    total_seq2seq_rows, : min(t + 1, sequence_length - 1), :
                ] = next_vitals[i, : min(t + 1, sequence_length - 1), :]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
                seq2seq_static_features[total_seq2seq_rows] = static_features[i]
                seq2seq_subject_ids[total_seq2seq_rows] = signal_id

                total_seq2seq_rows += 1

                # add modified sequence with previous treatment as current treatment
                seq2seq_active_entries[
                    total_seq2seq_rows, : (t + 1), :
                ] = active_entries[i, : (t + 1), :]
                if "stabilized_weights" in self.data:
                    seq2seq_stabilized_weights[
                        total_seq2seq_rows, : (t + 1)
                    ] = stabilized_weights[i, : (t + 1)]
                seq2seq_current_treatments[
                    total_seq2seq_rows, : (t + 1), :
                ] = current_treatments[i, : (t + 1), :]
                seq2seq_previous_treatments[
                    total_seq2seq_rows, : (t + 1), :
                ] = previous_treatments[i, : (t + 1), :]
                seq2seq_outputs[total_seq2seq_rows, : (t + 1), :] = outputs[
                    i, : (t + 1), :
                ]
                seq2seq_prev_outputs[total_seq2seq_rows, : (t + 1), :] = prev_outputs[
                    i, : (t + 1), :
                ]
                seq2seq_vitals[total_seq2seq_rows, : (t + 1), :] = vitals[
                    i, : (t + 1), :
                ]
                seq2seq_next_vitals[
                    total_seq2seq_rows, : min(t + 1, sequence_length - 1), :
                ] = next_vitals[i, : min(t + 1, sequence_length - 1), :]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
                seq2seq_static_features[total_seq2seq_rows] = static_features[i]
                seq2seq_subject_ids[total_seq2seq_rows] = signal_id

                seq2seq_current_treatments[
                    total_seq2seq_rows, prediction_index, :
                ] = seq2seq_current_treatments[
                    total_seq2seq_rows, prediction_index - 1, :
                ]
                seq2seq_previous_treatments[
                    total_seq2seq_rows, prediction_index, :
                ] = seq2seq_previous_treatments[
                    total_seq2seq_rows, prediction_index - 1, :
                ]

                total_seq2seq_rows += 1

        # Filter everything shorter
        seq2seq_previous_treatments = seq2seq_previous_treatments[
            :total_seq2seq_rows, :, :
        ]
        seq2seq_current_treatments = seq2seq_current_treatments[
            :total_seq2seq_rows, :, :
        ]
        seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
        seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
        seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
        seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
        seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
        seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

        if "stabilized_weights" in self.data:
            seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

        new_data = {
            "prev_treatments": seq2seq_previous_treatments,
            "current_treatments": seq2seq_current_treatments,
            "static_features": seq2seq_static_features,
            "prev_outputs": seq2seq_prev_outputs,
            "outputs": seq2seq_outputs,
            "vitals": seq2seq_vitals,
            "next_vitals": seq2seq_next_vitals,
            "unscaled_outputs": seq2seq_outputs * self.scaling_params["output_stds"]
            + self.scaling_params["output_means"],
            "sequence_lengths": seq2seq_sequence_lengths,
            "active_entries": seq2seq_active_entries,
            "subject_ids": seq2seq_subject_ids,
            # 'future_past_split': seq2seq_sequence_lengths - projection_horizon
            "future_past_split": seq2seq_sequence_lengths - time_shift - 1,
        }
        if "stabilized_weights" in self.data:
            new_data["stabilized_weights"] = seq2seq_stabilized_weights

        self.data = new_data


class MIMIC3RDDRealDatasetCollection(RealDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_f)
    """

    def __init__(
        self,
        path: str,
        min_seq_length: int = 30,
        max_seq_length: int = 60,
        seed: int = 100,
        max_number: int = None,
        split: dict = {"val": 0.2, "test": 0.2},
        projection_horizon: int = 5,
        autoregressive=True,
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

        # Train/val/test random_split
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

        if split["val"] > 0.0:
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
        else:
            static_features_train = static_features
            treatments_train, outcomes_train, vitals_train, outcomes_unscaled_train = (
                treatments,
                outcomes,
                vitals,
                outcomes_unscaled,
            )

        self.train_f = MIMIC3RDDRealDataset(
            treatments_train,
            outcomes_train,
            vitals_train,
            static_features_train,
            outcomes_unscaled_train,
            scaling_params,
            "train",
        )
        if split["val"] > 0.0:
            self.val_f = MIMIC3RDDRealDataset(
                treatments_val,
                outcomes_val,
                vitals_val,
                static_features_val,
                outcomes_unscaled_val,
                scaling_params,
                "val",
            )
        self.test_f = MIMIC3RDDRealDataset(
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

    def process_data_multi_train(self):
        """
        Used by CT
        """
        self.train_f_multi = deepcopy(self.train_f)

        # Multiplying test trajectories
        self.train_f_multi.explode_trajectories(self.projection_horizon)

        self.train_f_multi.process_sequential_test(self.projection_horizon)
        self.train_f_multi.process_sequential_multi(self.projection_horizon)
