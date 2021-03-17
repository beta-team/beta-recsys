"""isort:skip_file."""
import argparse
import math
import os
import sys
import time

sys.path.append("../")

import torch
from ray import tune
from torch.utils.data import DataLoader
from tqdm import tqdm

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.vbcar import VBCAREngine
from beta_rec.utils.common_util import DictToObject, str2bool
from beta_rec.utils.monitor import Monitor
from beta_rec.datasets.data_load import load_split_dataset
from beta_rec.data.grocery_data import GroceryData


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run VBCAR..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/vbcar_default.json",
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
    parser.add_argument("--root_dir", nargs="?", type=str, help="working directory")
    parser.add_argument(
        "--n_sample", nargs="?", type=int, help="Number of sampled triples."
    )
    parser.add_argument(
        "--tune",
        nargs="?",
        type=str2bool,
        help="Tune parameter",
    )
    parser.add_argument("--sub_set", nargs="?", type=int, help="Subset of dataset.")
    parser.add_argument(
        "--temp_train",
        nargs="?",
        type=int,
        help="IF value >0, then the model will be trained based on the temporal feeding, else use normal trainning.",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument(
        "--late_dim",
        nargs="?",
        type=int,
        help="Dimension of the latent layers.",
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Intial learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    parser.add_argument("--optimizer", nargs="?", type=str, help="OPTI")
    parser.add_argument("--activator", nargs="?", type=str, help="activator")
    parser.add_argument("--alpha", nargs="?", type=float, help="ALPHA")
    return parser.parse_args()


class VBCAR_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config):
        """Initialize VBCAR_train Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(VBCAR_train, self).__init__(self.config)

    def load_dataset(self):
        """Load dataset."""
        split_data = load_split_dataset(self.config)
        self.data = GroceryData(split_dataset=split_data, config=self.config)
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items

    def train(self):
        """Train the model."""
        self.load_dataset()
        self.train_data = self.data.sample_triple()
        self.config["model"]["alpha_step"] = (1 - self.config["model"]["alpha"]) / (
            self.config["model"]["max_epoch"]
        )
        self.config["user_fea"] = self.data.user_feature
        self.config["item_fea"] = self.data.item_feature
        self.engine = VBCAREngine(self.config)
        self.engine.data = self.data
        assert hasattr(self, "engine"), "Please specify the exact model engine !"
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        print("Start training... ")
        epoch_bar = tqdm(range(self.config["model"]["max_epoch"]), file=sys.stdout)
        self.max_n_update = self.config["model"]["max_n_update"]
        for epoch in epoch_bar:
            print(f"Epoch {epoch} starts !")
            print("-" * 80)
            if epoch > 0 and self.eval_engine.n_no_update == 0:
                # previous epoch have already obtained better result
                self.engine.save_checkpoint(
                    model_dir=os.path.join(
                        self.config["system"]["model_save_dir"], "model.cpk"
                    )
                )

            if self.eval_engine.n_no_update >= self.max_n_update:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        self.max_n_update
                    )
                )
                break
            data_loader = DataLoader(
                torch.LongTensor(self.train_data.to_numpy()).to(self.engine.device),
                batch_size=self.config["model"]["batch_size"],
                shuffle=True,
                drop_last=True,
            )
            self.engine.train_an_epoch(data_loader, epoch_id=epoch)
            self.eval_engine.train_eval(
                self.data.valid[0], self.data.test[0], self.engine.model, epoch
            )
            # anneal alpha
            self.engine.model.alpha = min(
                self.config["model"]["alpha"]
                + math.exp(epoch - self.config["model"]["max_epoch"] + 20),
                1,
            )
            """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
            lr = self.config["model"]["lr"] * (0.5 ** (epoch // 10))
            for param_group in self.engine.optimizer.param_groups:
                param_group["lr"] = lr
        self.config["run_time"] = self.monitor.stop()
        return self.eval_engine.best_valid_performance


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model
    """
    train_engine = VBCAR_train(DictToObject(config))
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
        train_engine = VBCAR_train(args)
        train_engine.tune(tune_train)
    else:
        print("Run application with single config ...")
        train_engine = VBCAR_train(args)
        train_engine.train()
        train_engine.test()
