import os
import time

import torch
from munch import munchify
from ray import tune
from torch.utils.data import DataLoader

from ..core.recommender import Recommender
from ..models.vbcar import VBCAREngine
from ..utils.monitor import Monitor


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    data = config["data"]
    train_engine = VBCAREngine(munchify(config))
    result = train_engine.train(data)
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.report(
        valid_metric=result["valid_metric"],
        model_save_dir=result["model_save_dir"],
    )


class VBCAR(Recommender):
    """The VBCAR Model."""

    def __init__(self, config):
        """Initialize the config of this recommender.

        Args:
            config:
        """
        super(VBCAR, self).__init__(config, name="VBCAR")

    def init_engine(self, data):
        """Initialize the required parameters for the model.

        Args:
            data: the Dataset object.

        """
        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = VBCAREngine(self.config)

    def train(self, data):
        """Training the model.

        Args:
            data: the Dataset object.

        Returns:
            dict: save k,v for "best_valid_performance" and "model_save_dir"

        """
        if ("tune" in self.args) and (self.args["tune"]):  # Tune the model.
            self.args.data = data
            tune_result = self.tune(tune_train)
            best_result = tune_result.loc[tune_result["valid_metric"].idxmax()]
            return {
                "valid_metric": best_result["valid_metric"],
                "model_save_dir": best_result["model_save_dir"],
            }

        self.gpu_id, self.config["device_str"] = self.get_device()  # Train the model.
        data.config = self.config
        data.init_item_fea()
        data.init_user_fea()
        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.config["user_fea"] = data.user_feature
        self.config["item_fea"] = data.item_feature
        self.engine = VBCAREngine(self.config)
        self.engine.data = data
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.train_data = data.sample_triple()
        train_loader = DataLoader(
            torch.LongTensor(self.train_data.to_numpy()).to(self.engine.device),
            batch_size=self.config["model"]["batch_size"],
            shuffle=True,
            drop_last=True,
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
