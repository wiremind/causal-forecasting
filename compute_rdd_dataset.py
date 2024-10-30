import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.rdd.rdd import compute_rdd_values_n_jobs
from src.rdd.utils import encode_treatments, load_mimic_raw_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path="./config/", version_base="1.3.2")
def main(args: DictConfig):
    """
    Computing the CATE dataset using RDD method
    :param args: arguments of run as DictConfig

    Returns: path of the saved dataframe
    """

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    # Format a dataframe from the raw data
    logger.info("Reading and formatting dataframe")
    df = load_mimic_raw_data(dataset_config=dict(args.dataset))
    df.dropna(inplace=True)  # We dion
    df = encode_treatments(df=df, treatment_column=args.rdd.treatment_column)

    # Compute the RDD dataset
    logger.info("Compute RDD dataset")
    rdd_df = compute_rdd_values_n_jobs(
        df=df,
        treatment_column=args.rdd.treatment_column,
        outcome_column=args.rdd.outcome_column,
        rdd_model_class_path=args.rdd.rdd_model_class_path,
        time_step_column=args.rdd.time_step_column,
        time_series_unique_id_columns=list(args.rdd.time_series_unique_id_columns),
        rdd_model_kwargs=dict(args.rdd.rdd_model_kwargs),
        static_columns_to_add=list(args.rdd.static_columns_to_add),
        n_jobs=args.rdd.n_jobs,
    )

    rdd_df.to_parquet(args.rdd.destination_file_path)
    logger.info(f"Saved dataframe at {args.rdd.destination_file_path}")

    return args.rdd.destination_file_path


if __name__ == "__main__":
    main()
