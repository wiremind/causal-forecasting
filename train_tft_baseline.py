import logging
import os

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.data.mimic_iii.real_dataset import MIMIC3RealDatasetCollection
from src.data.mimic_iii.tft_dataset import MIMIC3TFTRealDataset
from src.models.utils import set_seed
from src.rdd.utils import from_fully_qualified_import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path="./config/", version_base="1.3.2")
def main(args: DictConfig):

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda *args: sum(list(args)), replace=True)
    OmegaConf.register_new_resolver("len", len, replace=True)
    logger.info("\n" + OmegaConf.to_yaml(args, resolve=True))

    set_seed(args.exp.seed)

    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}", monitor="val_loss", mode="min"
    )
    model_class = from_fully_qualified_import(args.model._target_)
    model = model_class(**dict(args.model.params))

    dataset_collection = MIMIC3RealDatasetCollection(
        args.dataset.path,
        min_seq_length=args.dataset.min_seq_length,
        max_seq_length=args.dataset.max_seq_length,
        seed=args.exp.seed,
        max_number=args.dataset.max_number,
        split=args.dataset.split,
        projection_horizon=args.dataset.projection_horizon,
        autoregressive=args.dataset.autoregressive,
        outcome_list=args.dataset.outcome_list,
        vitals=args.dataset.vital_list,
        treatment_list=args.dataset.treatment_list,
        static_list=args.dataset.static_list,
        dataset_class=MIMIC3TFTRealDataset,
    )

    dataset_collection.process_data_multi_val()
    dataset_collection.process_data_multi_train()

    splitted_directory = args.model.destination_directory.split(os.path.sep)
    try:
        seed_idx = HydraConfig.get().job.num
    except MissingMandatoryValue:
        seed_idx = 0
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.exp.max_epochs,
        devices=args.exp.gpus,
        callbacks=checkpoint_callback,
        logger=TensorBoardLogger(
            save_dir=os.path.sep.join(splitted_directory[:-1]),
            name=splitted_directory[-1],
            version=f"{args.model.name}_{seed_idx}",
        ),
        deterministic=args.exp.deterministic,
    )

    train_loader = DataLoader(
        dataset_collection.train_f_multi,
        shuffle=True,
        batch_size=args.dataset.batch_size,
    )
    val_loader = DataLoader(dataset_collection.val_f_multi, batch_size=512)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
