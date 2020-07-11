# Introduction with Examples
We shortly introduce the fundamental components of Beta-RecSys through self-contained examples. 
At its core, Beta-RecSys provides the following main features:

## Load Build-in Datasets
The Beta-Recsys datasets package (beta_rec.datasets) provides users a wide range of datasets for recommendation system training. 
With one line codes, you can obtain the split dataset based on a json config, 
where that process actually consists of Downloading the raw data from a public link, Decompressing the raw data, 
Preprocessing the raw data into a standard interaction DataFrame object, and Splitting the dataset into 
Training/Validation/Testing sets. More details can be found [here](https://beta-recsys.readthedocs.io/en/latest/notes/datasets.html).
```python
from beta_rec.datasets.data_load import load_split_dataset
config = {
        "dataset": "ml_100k",
        "data_split": "leave_one_out",
        "download": False,
        "random": False,
        "test_rate": 0.2,
        "by_user": False,
        "n_test": 10,
        "n_negative": 100,
    }

data_split = load_split_dataset(config)
```
where data_split is a tuple consists of train, valid and test sets.

- train (DataFrame): Interaction for training.
- valid (DataFrame/list(DataFrame)): List of interactions for validation.
- test (DataFrame/list(DataFrame)): List of interactions for testing.

## Construct Dataloaders for training
The Beta-Recsys data package (beta_rec.data) provides the tools to further convert the split data sets into usable 
data structures (i.e. pytorch DataLoaders) (e.g. BPR (Bayesian personalized ranking) DataLoarder tensors with <user, positive_item, nagetive_item> or 
BCE (Binary Cross-Entropy) DataLoarder tensors <user, item, rating>), dependant on the requirements/supported features of the target model.

```python
from beta_rec.data.base_data import BaseData
data = BaseData(data_split)

# Instance a bpr DataLoader
train_loader = data.instance_bpr_loader(
                batch_size=512,
                device="cpu",
            )

# Instance a nce DataLoader
train_loader = data.instance_bce_loader(
                batch_size=512,
                device="cpu",
            )
```

## Run Matrix Factorization model
Beta-Recsys provides 9 recommendation models that can be used out-of-the-box.
For each model, we provide a default config/hyperparamter setting in the format if JOSN.
Note that these models also accept hyperparamters from command lines.
### Default config (configs/)

```json
{
    "system": {
        "root_dir": "../",
        "log_dir": "logs/",
        "result_dir": "results/",
        "process_dir": "processes/",
        "checkpoint_dir": "checkpoints/",
        "dataset_dir": "datasets/",
        "run_dir": "runs/",
        "tune_dir": "tune_results/",
        "device": "gpu",
        "seed": 2020,
        "metrics": ["ndcg", "precision", "recall", "map"],
        "k": [5,10,20],
        "valid_metric": "ndcg",
        "valid_k": 10,
        "result_file": "mf_result.csv"
    },
    "dataset": {
        "dataset": "ml_100k",
        "data_split": "leave_one_out",
        "download": false,
        "random": false,
        "test_rate": 0.2,
        "by_user": false,
        "n_test": 10,
        "n_negative": 100,
        "result_col": ["dataset","data_split","test_rate","n_negative"]
    },
    "model": {
        "model": "MF",
        "config_id": "default",
        "emb_dim": 64,
        "num_negative": 4,
        "batch_size": 400,
        "batch_eval": true,
        "dropout": 0.0,
        "optimizer": "adam",
        "loss": "bpr",
        "lr": 0.05,
        "reg": 0.001,
        "max_epoch": 20,
        "save_name": "mf.model",
        "result_col": ["model","emb_dim","batch_size","dropout","optimizer","loss","lr","reg"]
    }
}

```
## examples/train_mf.py 

```shell script
python train_mf.py 
```

## Tune hyper-parameters for the Matrix Factorization model
To support easier and faster hyperparameter tuning for each model, 
we also integrate the [Ray](https://github.com/ray-project/ray) framework, 
which is a Python library for model training at scale. 
This enables the distribution of model training/tuning across multiple gpus and/or compute nodes.
```json
{
"tunable": [
        {"name": "loss", "type": "choice", "values": ["bce", "bpr"]}
    ]
}
```
Then run train_mf.py with parameter '--tune'

```shell script
python train_mf.py --tune True
```

## Build a Matrix Factorization model with less codes

TBD