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


## Building your own model with the framework in 3 steps
> An example of the Matrix Factorization

### 1.  create mf.py in the models folder
```python
import torch
import torch.nn as nn
from torch.nn import Parameter

from beta_rec.models.torch_engine import ModelEngine
from beta_rec.utils.common_util import print_dict_as_table, timeit


class MF(torch.nn.Module):
    """A pytorch Module for Matrix Factorization."""

    def __init__(self, config):
        """Initialize MF Class."""
        super(MF, self).__init__()
        self.config = config
        self.device = self.config["device_str"]
        self.stddev = self.config["stddev"] if "stddev" in self.config else 0.1
        self.n_users = self.config["n_users"]
        self.n_items = self.config["n_items"]
        self.emb_dim = self.config["emb_dim"]
        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        self.global_bias = Parameter(torch.zeros(1))
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)
        self.global_bias.data.fill_(0.0)
        nn.init.normal_(self.user_emb.weight, 0, self.stddev)
        nn.init.normal_(self.item_emb.weight, 0, self.stddev)

    def forward(self, batch_data):
        """Trian the model.

        Args:
            batch_data: tuple consists of (users, pos_items, neg_items), which must be LongTensor.
        """
        users, items = batch_data
        u_emb = self.user_emb(users)
        u_bias = self.user_bias(users)
        i_emb = self.item_emb(items)
        i_bias = self.item_bias(items)
        scores = torch.sigmoid(
            torch.sum(torch.mul(u_emb, i_emb).squeeze(), dim=1)
            + u_bias.squeeze()
            + i_bias.squeeze()
            + self.global_bias
        )
        regularizer = (
            (u_emb ** 2).sum()
            + (i_emb ** 2).sum()
            + (u_bias ** 2).sum()
            + (i_bias ** 2).sum()
        ) / u_emb.size()[0]
        return scores, regularizer

    def predict(self, users, items):
        """Predcit result with the model.

        Args:
            users (int, or list of int):  user id(s).
            items (int, or list of int):  item id(s).
        Return:
            scores (int, or list of int): predicted scores of these user-item pairs.
        """
        users_t = torch.LongTensor(users).to(self.device)
        items_t = torch.LongTensor(items).to(self.device)
        with torch.no_grad():
            scores, _ = self.forward((users_t, items_t))
        return scores


class MFEngine(ModelEngine):
    """MFEngine Class."""

    def __init__(self, config):
        """Initialize MFEngine Class."""
        self.config = config
        print_dict_as_table(config["model"], tag="MF model config")
        self.model = MF(config["model"])
        self.reg = (
            config["model"]["reg"] if "reg" in config else 0.0
        )  # the regularization coefficient.
        self.batch_size = config["model"]["batch_size"]
        super(MFEngine, self).__init__(config)
        self.model.to(self.device)
        self.loss = (
            self.config["model"]["loss"] if "loss" in self.config["model"] else "bpr"
        )
        print(f"using {self.loss} loss...")

    def train_single_batch(self, batch_data):
        """Train a single batch.

        Args:
            batch_data (list): batch users, positive items and negative items.
        Return:
            loss (float): batch loss.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.optimizer.zero_grad()
        if self.loss == "bpr":
            users, pos_items, neg_items = batch_data
            pos_scores, pos_regularizer = self.model.forward((users, pos_items))
            neg_scores, neg_regularizer = self.model.forward((users, neg_items))
            loss = self.bpr_loss(pos_scores, neg_scores)
            regularizer = pos_regularizer + neg_regularizer
        elif self.loss == "bce":
            users, items, ratings = batch_data
            scores, regularizer = self.model.forward((users, items))
            loss = self.bce_loss(scores, ratings)
        else:
            raise RuntimeError(
                f"Unsupported loss type {self.loss}, try other options: 'bpr' or 'bce'"
            )
        batch_loss = loss + self.reg * regularizer
        batch_loss.backward()
        self.optimizer.step()
        return loss.item(), regularizer.item()

    @timeit
    def train_an_epoch(self, train_loader, epoch_id):
        """Train a epoch, generate batch_data from data_loader, and call train_single_batch.

        Args:
            train_loader (DataLoader):
            epoch_id (int): the number of epoch.
        """
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0.0
        regularizer = 0.0
        for batch_data in train_loader:
            loss, reg = self.train_single_batch(batch_data)
            total_loss += loss
            regularizer += reg
        print(f"[Training Epoch {epoch_id}], Loss {loss}, Regularizer {regularizer}")
        self.writer.add_scalar("model/loss", total_loss, epoch_id)
        self.writer.add_scalar("model/regularizer", regularizer, epoch_id)
```

In the mf.py, you may want to add two classes, class **NEWMODEL** (all in capital) and class **NEWMODELEngine**. The NEWMODEL calss should include all necessary initialisations (e.g. embeddings initialisation), *forward function* to calculate all intermedinate variables and *predict function* to calculate predicted scores for each (user, item) pair. In the NEWMODELEngine, first you need load the training data and corresponding configs. Then you use two functions *train_an_epoch* and *train_single_batch* to feed data to the **NEWMODEL** class. A classic train_loader, which can sample user, positive items and negative items is already included in our project. You can see much efforts by loading existing functions.
### 2.  create mf_default.json in the configs folder

You also need a .json file, which includes all parameters for your models. This config file bring much convenience when you want to run a model several times with different parameters. Parameters can be changed from the command line. Below is a exmaple of a config file for the matrix factorisation model.
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
    },
    "tunable": [
        {"name": "loss", "type": "choice", "values": ["bce", "bpr"]}
    ]
}
```
### 3.  create new_example.py in the examples folder
```python
import argparse
import os
import sys
import time

sys.path.append("../")

from ray import tune

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.mf import MFEngine
from beta_rec.utils.common_util import DictToObject, str2bool
from beta_rec.utils.monitor import Monitor


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run MF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/mf_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    parser.add_argument(
        "--root_dir", nargs="?", type=str, help="Root path of the project",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        help="Options are: tafeng, dunnhunmby and instacart",
    )
    parser.add_argument(
        "--data_split",
        nargs="?",
        type=str,
        help="Options are: leave_one_out and temporal",
    )
    parser.add_argument(
        "--tune", nargs="?", type=str2bool, help="Tun parameter",
    )
    parser.add_argument(
        "--device", nargs="?", type=str, help="Device",
    )
    parser.add_argument(
        "--loss", nargs="?", type=str, help="loss: bpr or bce",
    )
    parser.add_argument(
        "--remark", nargs="?", type=str, help="remark",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initial learning rate.")
    parser.add_argument("--reg", nargs="?", type=float, help="regularization.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


class MF_train(TrainEngine):
    """MF_train Class."""

    def __init__(self, args):
        """Initialize MF_train Class."""
        print(args)
        super(MF_train, self).__init__(args)

    def train(self):
        """Train the model."""
        self.load_dataset()
        self.gpu_id, self.config["device_str"] = self.get_device()
        """ Main training navigator

        Returns:

        """
        # Train NeuMF without pre-train
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        if self.config["model"]["loss"] == "bpr":
            train_loader = self.data.instance_bpr_loader(
                batch_size=self.config["model"]["batch_size"],
                device=self.config["model"]["device_str"],
            )
        elif self.config["model"]["loss"] == "bce":
            train_loader = self.data.instance_bce_loader(
                num_negative=self.config["model"]["num_negative"],
                batch_size=self.config["model"]["batch_size"],
                device=self.config["model"]["device_str"],
            )
        else:
            raise ValueError(
                f"Unsupported loss type {self.config['loss']}, try other options: 'bpr' or 'bce'"
            )

        self.engine = MFEngine(self.config)
        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self._train(self.engine, train_loader, self.model_save_dir)
        self.config["run_time"] = self.monitor.stop()
        return self.eval_engine.best_valid_performance


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    train_engine = MF_train(DictToObject(config))
    best_performance = train_engine.train()
    train_engine.test()
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.track.log(valid_metric=best_performance)


if __name__ == "__main__":
    args = parse_args()
    if args.tune:
        train_engine = MF_train(args)
        train_engine.tune(tune_train)
    else:
        train_engine = MF_train(args)
        train_engine.train()
        train_engine.test()

```

In this new_example.py file, you need import the TrainEngine from core and the NEWMODELEngine from the new_model.py. The parse_args function will help you to load parameters from the command line and the config file. You can simply run your model once or you may want to apply a grid search by the Tune module. You should define all tunable parameters in your config file.
