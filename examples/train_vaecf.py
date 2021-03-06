"""isort:skip_file."""
import argparse
import os
import sys
import time

sys.path.append("../")

from ray import tune
from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.vae_recom import VAECFEngine
from beta_rec.utils.common_util import DictToObject, str2bool
from beta_rec.utils.monitor import Monitor


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run VAECF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/vaecf_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    parser.add_argument(
        "--root_dir",
        nargs="?",
        type=str,
        help="Root path of the project",
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
        "--tune", nargs="?", default=False, type=str2bool, help="Tun parameter",
    )
    parser.add_argument(
        "--device", nargs="?", type=str, help="Device",
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


class VAECF_train(TrainEngine):
    """MF_train Class."""

    def __init__(self, args):
        """Initialize MF_train Class."""
        print(args)
        super(VAECF_train, self).__init__(args)

    def train(self):
        """Train the model."""
        self.load_dataset()
        self.gpu_id, self.config["device_str"] = self.get_device()
        """ Main training navigator

        Returns:

        """

        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        train_loader = self.data.instance_vae_loader(
            batch_size=self.config["model"]["batch_size"],
            device=self.config["model"]["device_str"],
        )

        self.config["model"]["n_items"] = self.data.n_items
        self.config["model"]["n_users"] = self.data.n_users
        self.engine = VAECFEngine(self.config)
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
    train_engine = VAECFEngine(DictToObject(config))
    best_performance = train_engine.train()
    train_engine.test()
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.track.log(valid_metric=best_performance)


if __name__ == "__main__":
    args = parse_args()
    if args.tune:
        train_engine = VAECF_train(args)
        train_engine.tune(tune_train)
    else:
        train_engine = VAECF_train(args)
        train_engine.train()
        train_engine.test()
