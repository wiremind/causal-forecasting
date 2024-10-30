from abc import abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge


class AbstractRddModel:
    def __init__(self, **kwargs: dict[str, Any]) -> None:
        self.model: RegressorMixin | None = None

    @abstractmethod
    def fit(
        self,
        sorted_single_series_df: pd.DataFrame,
        outcome_column: str,
        time_step_column: str,
        left_time_step_min: int,
        left_time_step_max: int,
        right_time_step_min: int,
        right_time_step_max: int,
        prediction_time_step: int,
        compute_confidence: bool = False,
    ) -> "AbstractRddModel | None":
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, sorted_single_series_df: pd.DataFrame, time_steps: list[int]
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def estimate_cate(self) -> float:
        raise NotImplementedError


class SingleLinearRddModel(AbstractRddModel):
    def __init__(
        self,
        kernel_bandwidth: int = 5,
        kernel_shape: Literal["rectangular", "triangle"] = "rectangular",
    ) -> None:
        super().__init__()
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_shape = kernel_shape
        self._switching_time_step = 0

    def compute_sample_weights(
        self, time_step_values: np.ndarray, prediction_time_step: int
    ) -> np.ndarray | None:
        sample_weights = None
        if len(time_step_values) > 0:
            sample_weights = (
                np.abs(time_step_values - prediction_time_step) <= self.kernel_bandwidth
            ).astype(float)

            if self.kernel_shape == "triangle":
                sample_weights *= (
                    1
                    - np.abs(time_step_values - prediction_time_step)
                    / self.kernel_bandwidth
                )

        return sample_weights

    def generate_features(self, time_step_values: list[int]) -> pd.DataFrame:
        # time steps are centered towards 0 to easily differentiate model parameters associated to the left
        # part or the right part
        time_step_values = np.array(time_step_values) - self._switching_time_step
        features_df = pd.DataFrame(
            {
                "time_step": time_step_values,  # this feature is used to estimate the slope of the left part
                "right_price_indicator": (time_step_values > 0).astype(
                    int
                ),  # this feature is used to estimate
                # the intercept of right part.
            }
        )
        # this feature is used to estimate the slope of the right part
        features_df["right_treatment_indicator_time_step"] = (
            features_df["time_step"] * features_df["right_price_indicator"]
        )

        return features_df

    def fit(
        self,
        sorted_single_series_df: pd.DataFrame,
        time_step_column: str,
        outcome_column: str,
        left_time_step_min: int,
        left_time_step_max: int,
        right_time_step_min: int,
        right_time_step_max: int,
        prediction_time_step: int,
        compute_confidence: bool = False,
        # retirer le principe des index et paser aux JX direct
    ) -> "AbstractRddModel | None":
        self._switching_time_step = prediction_time_step
        single_series_df_fit_indexes = sorted_single_series_df.query(
            f"@left_time_step_min <= {time_step_column} <= @right_time_step_max"
        ).index.values

        model_time_step_values = sorted_single_series_df.loc[
            single_series_df_fit_indexes, time_step_column
        ].values

        # compute sample weights
        sample_weights = self.compute_sample_weights(
            time_step_values=model_time_step_values[
                model_time_step_values != self._switching_time_step
            ],
            prediction_time_step=self._switching_time_step,
        )
        if sample_weights is not None and sample_weights.max() == 0:
            return None

        # generate features
        fit_df = (
            sorted_single_series_df[[time_step_column, outcome_column]]
            .loc[single_series_df_fit_indexes]
            .query(f"{time_step_column} != @self._switching_time_step")
        )
        outcome_values = fit_df[outcome_column].values
        features_df = self.generate_features(fit_df[time_step_column].values)
        # fit model
        self.model = Ridge().fit(features_df, outcome_values, sample_weights)

        return self

    def predict(
        self, sorted_single_series_df: pd.DataFrame, time_steps: list[int]
    ) -> list[float]:
        if self.model is None:
            raise Exception("Can't predict if the model is not fitted")

        return self.model.predict(
            self.generate_features(time_step_values=time_steps)
        ).tolist()

    def estimate_cate(self) -> float:
        return self.model.coef_[1]
