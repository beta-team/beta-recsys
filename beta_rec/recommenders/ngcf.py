import os
import time

import numpy as np
import torch
from munch import munchify
from ray import tune

from ..core.recommender import Recommender
from ..data.deprecated_data_base import DataLoaderBase
from ..models.ngcf import NGCFEngine
from ..utils.monitor import Monitor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    data = config["data"]
    train_engine = NGCF(munchify(config))
    result = train_engine.train(data)
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.report(
        valid_metric=result["valid_metric"],
        model_save_dir=result["model_save_dir"],
    )


class NGCF(Recommender):
    """The NGCF Model."""

    def __init__(self, config):
        """Initialize the config of this recommender.

        Args:
            config:
        """
        super(NGCF, self).__init__(config, name="NGCF")

    def init_engine(self, data):
        """Initialize the required parameters for the model.

        Args:
            data: the Dataset object.

        """
        self.sample_generator = DataLoaderBase(ratings=data.train)
        adj_mat, norm_adj_mat, mean_adj_mat = self.sample_generator.get_adj_mat(
            self.config
        )
        norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj_mat)

        self.config["model"]["norm_adj"] = norm_adj

        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = NGCFEngine(self.config)

    def train(self, data):
        """Training the model.

        Args:
            data: the Dataset object.

        Returns:
            dict: save k,v for "best_valid_performance" and "model_save_dir"

        """
        if ("tune" in self.args) and (self.args["tune"]):  # Tune the model.
            self.args.data = data
            return self.tune(tune_train)

        self.gpu_id, self.config["device_str"] = self.get_device()  # Train the model.

        self.sample_generator = DataLoaderBase(ratings=data.train)
        adj_mat, norm_adj_mat, mean_adj_mat = self.sample_generator.get_adj_mat(
            self.config
        )
        norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj_mat)

        self.config["model"]["norm_adj"] = norm_adj

        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = NGCFEngine(self.config)

        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        train_loader = data.instance_bpr_loader(
            batch_size=self.config["model"]["batch_size"],
            device=self.config["model"]["device_str"],
        )

        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self._train(
            self.engine,
            train_loader,
            self.model_save_dir,
            valid_df=data.valid[0],
            test_df=data.test[0],
        )
        self.config["run_time"] = self.monitor.stop()
        return {
            "valid_metric": self.eval_engine.best_valid_performance,
            "model_save_dir": self.model_save_dir,
        }
