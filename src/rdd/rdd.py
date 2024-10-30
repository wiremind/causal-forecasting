from collections.abc import Generator

import joblib
import pandas as pd
from tqdm.auto import tqdm

from src.rdd.rdd_models import AbstractRddModel
from src.rdd.utils import encode_treatments, from_fully_qualified_import


def _extract_switching_time_steps(
    sorted_single_series_df: pd.DataFrame, treatment_column: str
) -> list[tuple[int, int]]:
    """
    Extract all the time steps where the treatment changed in a single time series. For each treatment changed, this
    function returns the first time step index and the last time step index + 1 for this treatment

    :param sorted_single_series_df: dataframe containing data associated to a single time series sorted by time step
    :param treatment_column: column representing a unique value per treatment
    :return: a list of tuples composed of the first time step index and the last time step index + 1 for switching
    time step in this time series
    """
    treatments = sorted_single_series_df[treatment_column].values.tolist()
    contiguous_treatment_indexes = []

    current_treatment_first_index = 0
    for i in range(len(treatments)):
        if treatments[i] != treatments[current_treatment_first_index]:
            contiguous_treatment_indexes.append((current_treatment_first_index, i))
            current_treatment_first_index = i

    if current_treatment_first_index != len(treatments) - 1:
        contiguous_treatment_indexes.append(
            (current_treatment_first_index, len(treatments) - 1)
        )

    return contiguous_treatment_indexes


def rdd_indexes_iterator(
    constant_treatments_indexes: list[tuple[int, int]]
) -> Generator[tuple, None, None]:
    """
    generator iterating over all time series' switching time step. For switching time step, we extract
    the first time step and last time step index from the sorted single time series dataframe.
    The generator also skips switching time step where we don't have enough values to estimate the
    treatment effect using RDD.
    """
    for sequence_index, (time_step_min_index, time_step_max_index) in enumerate(
        constant_treatments_indexes[:-1]
    ):
        (
            next_time_step_min_index,
            next_time_step_max_index,
        ) = constant_treatments_indexes[sequence_index + 1]
        if (
            next_time_step_max_index <= next_time_step_min_index + 1
            or time_step_max_index <= time_step_min_index + 1
        ):
            # If we only have one day before the switching time step or after the switching time step we can't fit an
            # rdd model to predict the treatment effect
            continue

        # The first switching time step can't be used in RDD as it is a time step that had at least 2
        # different treatments set because of sampling
        next_time_step_min_index += 1
        time_step_min_index += 1
        prediction_index = time_step_max_index

        yield (
            time_step_min_index,
            time_step_max_index,
            next_time_step_min_index,
            next_time_step_max_index,
            prediction_index,
        )


def compute_subject_id_swicthing_time_steps_mapping(
    df: pd.DataFrame,
) -> dict[int, list[tuple[int, int]]]:
    """Computes a dict mapping eahc subject id to the list of switching time steps associated to it

    :param df: Raw dataframe

    Returns:
        dict[int, list[tuple[int, int]]]: mapping subject id to the switching time steps
    """
    formatted_df = df.sort_values(["subject_id", "hours_in"], ignore_index=True)
    formatted_df = encode_treatments(formatted_df, treatment_column="treatment")
    dict_mapping = {}

    def _fill_dict_mapping(_sorted_series_df, _dict_mapping):
        subject_id = _sorted_series_df["subject_id"].values[0]
        _dict_mapping[subject_id] = _extract_switching_time_steps(
            _sorted_series_df, treatment_column="treatment"
        )

    formatted_df.groupby("subject_id").apply(_fill_dict_mapping, dict_mapping)

    return dict_mapping


def compute_time_series_rdd_values(
    sorted_single_series_df: pd.DataFrame,
    rdd_model_class: type[AbstractRddModel],
    treatment_column: str,
    outcome_column: str,
    time_step_column: str,
    rdd_model_kwargs: dict | None = None,
    static_columns_to_add: list[str] | None = None,
) -> pd.DataFrame:
    """
    This function takes a single time series dataframe as an input, extract the witching time steps and computes, for
    each extracted switching time step, the CATE value using an RDD model.

    :param sorted_single_series_df: sorted_single_series_df dataframe sorted by time steps
    :param rdd_model_class: Simple model to use in order to extract the treatment effect
    :param treatment_column: treatment column. The time series dataframe should contain a unique value for each possible
    treatment value
    :param outcome_column: outcome column
    :param time_step_column: column representing the time step in the time series dataframe
    :param rdd_model_kwargs: kwargs passed to the RddModel class
    :param static_columns_to_add: additional columns to add into the output dataframe from the time series dataframe
    in order to do further data analysis

    :return: return a dataframe containing one row per switching time step observed in this time series. For each
    switching time step, we associate a CATE value using the RDD model
    """
    static_columns_to_add = static_columns_to_add or []
    res_df_dict: dict[str, list] = {
        time_step_column: [],
        "number_steps_left": [],
        "number_steps_right": [],
        "CATE": [],
        "left_treatment": [],
        "right_treatment": [],
    }
    res_df_dict.update({col: [] for col in static_columns_to_add})
    treatment_values = sorted_single_series_df[treatment_column].values.tolist()
    time_step_values = sorted_single_series_df[time_step_column].values.tolist()
    time_series_static_values = sorted_single_series_df.iloc[0]

    # We need to first extract all the switching time steps indexes in order to, given a switching time step index,
    # check the values of the next switching time step index
    constant_treatments_indexes = _extract_switching_time_steps(
        sorted_single_series_df=sorted_single_series_df,
        treatment_column=treatment_column,
    )

    # iterate over all switching time steps
    for (
        left_time_step_min_index,
        left_time_step_max_index,
        right_time_step_min_index,
        right_time_step_max_index,
        prediction_index,
    ) in rdd_indexes_iterator(
        constant_treatments_indexes=constant_treatments_indexes,
    ):
        # fit the rdd model on left and right time steps then do the prediction
        # for both the left treatment and right treatment at the prediction time step
        model = rdd_model_class(**(rdd_model_kwargs or {})).fit(
            sorted_single_series_df=sorted_single_series_df,
            left_time_step_min=time_step_values[left_time_step_min_index],
            left_time_step_max=time_step_values[left_time_step_max_index] - 1,
            right_time_step_min=time_step_values[right_time_step_min_index],
            right_time_step_max=time_step_values[right_time_step_max_index] - 1,
            prediction_time_step=time_step_values[prediction_index],
            outcome_column=outcome_column,
            time_step_column=time_step_column,
        )

        if model is None:
            # The data sent to fit the model might not be good enough to fit a relevant model
            continue

        cate = model.estimate_cate()

        # format results
        res_df_dict[time_step_column].append(time_step_values[prediction_index])
        res_df_dict["CATE"].append(cate)
        res_df_dict["number_steps_left"].append(
            left_time_step_max_index - left_time_step_min_index
        )
        res_df_dict["number_steps_right"].append(
            right_time_step_max_index - right_time_step_min_index
        )
        res_df_dict["left_treatment"].append(treatment_values[left_time_step_min_index])
        res_df_dict["right_treatment"].append(
            treatment_values[right_time_step_min_index]
        )
        for col in static_columns_to_add:
            res_df_dict[col].append(time_series_static_values[col])

    return pd.DataFrame(res_df_dict)


def compute_rdd_values(
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    rdd_model_class_path: str,
    time_step_column: str,
    time_series_unique_id_columns: list[str],
    rdd_model_kwargs: dict | None = None,
    static_columns_to_add: list[str] | None = None,
    str_index: str | None = None,
) -> pd.DataFrame:
    """
    Estimate CATE values using RDD model from a dataset of multiple time series

    :param df: original dataframe containing multiple time series
    :param treatment_column: treatment column name in df
    :param time_step_column: column representing the time step in the time series dataframe
    :param outcome_column: outcome column name in df
    :param rdd_model_class_path: class path of the rdd model to use in order to extract the elasticity value from
    the raw data
    :param time_series_unique_id_columns: columns defining the unique id of a time series inside df
    :param rdd_model_kwargs: kwargs passed to the rdd model class init
    :param str_index: unique index used to display current name in progress bar
    :param static_columns_to_add: additional columns to add into the output dataframe from the time series dataframe
    in order to do further data analysis

    :return: Return a dataframe containing on row per time series unique id/switching time step.
    Each row contains the rdd demand values, elasticity values and the other rdd metadata
    """
    rdd_model_class = from_fully_qualified_import(rdd_model_class_path)
    df.sort_values(
        [*time_series_unique_id_columns, time_step_column],
        inplace=True,
        ignore_index=True,
    )

    tqdm.pandas(desc=f"Estimating CATE per time series for index {str_index or 0}")
    return (
        df.groupby(time_series_unique_id_columns)
        .progress_apply(
            compute_time_series_rdd_values,
            rdd_model_class=rdd_model_class,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            time_step_column=time_step_column,
            rdd_model_kwargs=rdd_model_kwargs,
            static_columns_to_add=static_columns_to_add,
        )
        .reset_index(level=1, drop=True)
        .reset_index()
    )


def compute_rdd_values_n_jobs(
    df: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    rdd_model_class_path: str,
    time_step_column: str,
    time_series_unique_id_columns: list[str],
    rdd_model_kwargs: dict | None = None,
    static_columns_to_add: list[str] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute RDD dataset using multiple jobs.

    :param df: original dataframe containing multiple time series
    :param treatment_column: treatment column name in df
    :param time_step_column: column representing the time step in the time series dataframe
    :param outcome_column: outcome column name in df
    :param rdd_model_class_path: class path of the rdd model to use in order to extract the elasticity value from
    the raw data
    :param time_series_unique_id_columns: columns defining the unique id of a time series inside df
    :param n_jobs: n_jobs used to compute cate values. The dataframe is first split per
    chunk of time_series_unique_id_columns then each job compute the CATE associated to the time series in the chunk
    :param rdd_model_kwargs: kwargs passed to the rdd model class init
    :param static_columns_to_add: additional columns to add into the output dataframe from the time series dataframe
    in order to do further data analysis

    :return: Return a dataframe containing on row per time series unique id/switching time step.
    Each row contains the rdd demand values, elasticity values and the other rdd metadata
    """
    unique_id_column, time_series_ids_chunks = _split_time_series_in_chunks(
        df=df,
        n_chunks=n_jobs,
        time_series_unique_id_columns=time_series_unique_id_columns,
    )

    # Compute rdd dataset
    results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(compute_rdd_values)(
            df=df[df[unique_id_column].isin(set(ids_chunk))],
            rdd_model_class_path=rdd_model_class_path,
            rdd_model_kwargs=rdd_model_kwargs,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            time_step_column=time_step_column,
            time_series_unique_id_columns=time_series_unique_id_columns,
            static_columns_to_add=static_columns_to_add,
            str_index=str(idx),
        )
        for idx, ids_chunk in tqdm(
            enumerate(time_series_ids_chunks),
            desc="Computing arguments dataframe",
            total=len(time_series_ids_chunks),
        )
    )

    return pd.concat(results, axis=0)


def _split_time_series_in_chunks(
    df: pd.DataFrame, n_chunks: int, time_series_unique_id_columns: list[str]
) -> tuple[str, list[list[str]]]:
    """
    Create chunks of time series unique ids in order to split the process of the original dataframe df
    in multiple chunks.
    This function also create a colum to store the unique id of a time series if len(time_series_unique_id_columns) > 1
    :param df: original dataframe containing multiple time series
    :param n_chunks: number of chunks
    :pram time_series_unique_id_columns: columns representing a unique time series in df
    """
    if len(time_series_unique_id_columns) == 1:
        unique_id_col = time_series_unique_id_columns[0]
    else:
        unique_id_col = "time_series_unique_id"
        df[unique_id_col] = df[time_series_unique_id_columns[0]].astype(str)
        for col in time_series_unique_id_columns[1:]:
            df[unique_id_col] = df[unique_id_col] + "__" + df[col].astyupe(str)

    time_series_ids_chunks = []
    time_series_unique_ids = df[unique_id_col].unique().tolist()
    n_time_series = len(time_series_unique_ids)
    time_series_per_chunk = n_time_series // n_chunks
    time_series_to_add = n_time_series % n_chunks
    for chunk_idx in range(n_chunks):
        chunk_ids = time_series_unique_ids[
            chunk_idx * time_series_per_chunk : (chunk_idx + 1) * time_series_per_chunk
        ]
        if (chunk_idx + 1) <= time_series_to_add:
            chunk_ids.append(time_series_unique_ids[-(chunk_idx + 1)])
        time_series_ids_chunks.append(chunk_ids)

    return unique_id_col, time_series_ids_chunks
