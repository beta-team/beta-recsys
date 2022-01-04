"""isort:skip_file."""
import argparse
import os
import sys

sys.path.append("../")

import numpy as np
import torch

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.sgl import SGLEngine
from beta_rec.utils.monitor import Monitor


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run SGL..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/sgl_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument(
        "--tune",
        nargs="?",
        type=str,
        default=False,
        help="Tun parameter",
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")

    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col])
    return torch.sparse_coo_tensor(torch.tensor(indices), coo.data, coo.shape)


def _convert_csr_to_sparse_tensor_inputs(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col])
    return indices, coo.data, coo.shape


class SGL_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config):
        """Initialize SGL_train Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(SGL_train, self).__init__(config)
        self.load_dataset()
        self.build_data_loader()
        self.engine = SGLEngine(self.config)

    def build_data_loader(self):
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items
        norm_adj = self.data.create_sgl_mat(self.config)
        self.config["model"]["norm_adj"] = norm_adj

    def train(self):
        """Train the model."""
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self.engine = SGLEngine(self.config)
        train_loader = self.data.instance_bpr_loader(
            batch_size=self.config["model"]["batch_size"],
            device=self.config["model"]["device_str"],
        )
        self._train(self.engine, train_loader, self.model_save_dir)
        self.config["run_time"] = self.monitor.stop()
        return self.eval_engine.best_valid_performance


if __name__ == "__main__":
    args = parse_args()
    train_engine = SGL_train(args)

    train_engine.train()
    train_engine.test()
