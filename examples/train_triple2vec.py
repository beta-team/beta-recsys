"""isort:skip_file."""
import argparse
import os
import sys
import time

sys.path.append("../")

import torch
from ray import tune
from torch.utils.data import DataLoader

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.triple2vec import Triple2vecEngine
from beta_rec.utils.common_util import DictToObject, str2bool
from beta_rec.utils.monitor import Monitor
from beta_rec.datasets.data_load import load_split_dataset
from beta_rec.data.grocery_data import GroceryData


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run Triple2vec..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/triple2vec_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
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
        "--root_dir",
        nargs="?",
        type=str,
        help="working directory",
    )
    parser.add_argument(
        "--tune",
        nargs="?",
        type=str2bool,
        help="Tune parameters",
    )
    parser.add_argument(
        "--n_sample", nargs="?", type=int, help="Number of sampled triples."
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    parser.add_argument("--optimizer", nargs="?", type=str, help="OPTI")
    return parser.parse_args()


class Triple2vec_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config):
        """Init Triple2vec_train Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(Triple2vec_train, self).__init__(self.config)
        self.gpu_id, self.config["device_str"] = self.get_device()

    def load_dataset(self):
        """Load dataset."""
        split_data = load_split_dataset(self.config)
        self.data = GroceryData(split_dataset=split_data, config=self.config)
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items

    def train(self):
        """Train the model."""
        self.load_dataset()
        self.engine = Triple2vecEngine(self.config)
        self.engine.data = self.data
        self.train_data = self.data.sample_triple()
        train_loader = DataLoader(
            torch.LongTensor(self.train_data.to_numpy()).to(self.engine.device),
            batch_size=self.config["model"]["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
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
    train_engine = Triple2vec_train(DictToObject(config))
    best_performance = train_engine.train()
    tune.track.log(valid_metric=best_performance)
    train_engine.test()
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.tune:
        print("Start tune hyper-parameters ...")
        train_engine = Triple2vec_train(args)
        train_engine.tune(tune_train)
    else:
        print("Run application with single config ...")
        train_engine = Triple2vec_train(args)
        train_engine.train()
        train_engine.test()
