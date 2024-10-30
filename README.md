Training and evaluating causal forecasting models for time-series
==============================
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2204.07258-b31b1b.svg)](https://arxiv.org/abs/2204.07258)
[![Python application](https://github.com/Valentyn1997/CausalTransformer/actions/workflows/python-app.yml/badge.svg)](https://github.com/Valentyn1997/CausalTransformer/actions/workflows/python-app.yml) -->

This repository is a fork from the [CausalTransformer](https://github.com/Valentyn1997/CausalTransformer) repository. The forked repository normalize the data processing pipeline and provide the CausalTransofrmer model architecture.

The project is built with following Python libraries:
1. [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - deep learning models
2. [Hydra](https://hydra.cc/docs/intro/) - simplified command line arguments management
3. [Encodec](https://github.com/facebookresearch/encodec) - causal convolution

### Installations
First one needs to create a Python 3.11 virtual environment and install all the requirements:
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

As the causal transformer repository uses lower version of Pytorch-Lightning, we also created an other 
virtual environment to train the CT model using Python 3.9 and the requirements defined in `requirements-ct.txt`.

## Tasks

There are 5 different scripts each one of them for a specific task (either training, computing metrics or computing RDD dataset). All the script are based on the main config file `config/config.yaml`. Each task then use 
a subsection fo the config located in `configs/` folder.

### RDD Dataset

It is mandatory to first compute the RDD dataset before computing the RDD RMSE. The script associated to this task is `compute_rdd_dataset.py`. This script uses both the `dataset`section fo the config and the `rdd` section fo the config. The parameters used to reproduce the metrics are located at `config/rdd/mimic3_rdd.yaml`. You can run the script with `python compute_rdd_dataset.py`. This should create a dataset at `data/processed/rdd_dataset.parquet`.

### Model training
There are 2 types of TFT models:
- Baseline model is a plane deep learning architecture
- Causal model is comopsed of 3 sub models $m_0$/$e_0$/$\Theta$
All the code associated to those models are located at `src/models/causal_tft`

#### Training baseline model

The baseline model can be trained using the command `python train_tft_baseline.py +model=baseline`. The parameters associated to this training are associated to the `model`'s section in the config. The parameters we used to train the model are located at `config/model/baseline.yaml`. You can train multiple seeds of the model at the same time using `python train_tft_baseline.py --multirun +model=baseline exp.seed=10,101,1001,10010,10110`. The seeds `10,101,1001,10010,10110` are the one we used for our experiments. 

#### Training causal model

We propose 2 encoding for the causal model. The `one_hot` model encode the treatments using one hot encoding and the `cumulative` model uses cumulative sum. Models can respectively trained with the commands `python train_tft_causal.py --multirun +model=one_hot exp.seed=10,101,1001,10010,10110` and
`python train_tft_causal.py --multirun +model=cumulative exp.seed=10,101,1001,10010,10110`

#### Training CT model

In order to compare our model we decided to train a Causal Transformer model aswell. Based on the original repository, you can train a CT model using the command `python train_multi.py --multirun +backbone=ct +backbone/ct_hparams/mimic3_real=diastolic_blood_pressure +dataset=mimic3_real exp.seed=10,101,1001,10010,10110`. The model checkpoints should be saved under the multirun folder.


### Model evaluation

Before computing the metrics, you need to fill the config with the path of the trained model in `config/rdd/mimic3_rdd.yaml` under the `metrics` section.

#### Forecast metrics

You can compute the forecast metrics using the command `python compute_metrics.py`. This will create a json file under `data/processed/forecast_metrics.json`. This json file contain a section `paper_metrics_per_time_shift` with all the final metrics.

#### RDD metrics

You first need to compute the RDD dataset and fill the config at `config/rdd/mimic3_rdd.yaml`. Than you can compute the RDD metrics with the command `python compute_rdd_rmse.py`. This will create a json file under `data/processed/rdd_metrics.json`. This json file contain a section `paper_metrics_per_time_shift` with all the final metrics.

## Datasets
Before running any task, place MIMIC-III-extract dataset ([all_hourly_data.h5](https://github.com/MLforHealth/MIMIC_Extract)) to `data/processed/`

