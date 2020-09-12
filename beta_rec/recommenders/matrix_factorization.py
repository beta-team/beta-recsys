import os
import time

from beta_rec.models.mf import MFEngine
from beta_rec.recommenders import Recommender
from beta_rec.utils.monitor import Monitor
from munch import munchify
from ray import tune


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    data = config["data"]
    train_engine = MatrixFactorization(munchify(config))
    result = train_engine.train(data)
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.track.log(
        valid_metric=result["valid_metric"], model_save_dir=result["model_save_dir"],
    )


class MatrixFactorization(Recommender):
    def __init__(self, config):
        super(MatrixFactorization, self).__init__(config, name="MF")

    def init_engine(self, data):
        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = MFEngine(self.config)

    def train(self, data):
        """Main training navigator

        Returns:
            dict: save k,v for "best_valid_performance" and "model_save_dir"
        """

        """Tune the model."""
        if ("tune" in self.args) and (self.args["tune"]):
            self.args.data = data
            return self.tune(tune_train)

        """Train the model."""
        self.gpu_id, self.config["device_str"] = self.get_device()
        # Train NeuMF without pre-train

        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = MFEngine(self.config)

        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        if self.config["model"]["loss"] == "bpr":
            train_loader = data.instance_bpr_loader(
                batch_size=self.config["model"]["batch_size"],
                device=self.config["model"]["device_str"],
            )
        elif self.config["model"]["loss"] == "bce":
            train_loader = data.instance_bce_loader(
                num_negative=self.config["model"]["num_negative"],
                batch_size=self.config["model"]["batch_size"],
                device=self.config["model"]["device_str"],
            )
        else:
            raise ValueError(
                f"Unsupported loss type {self.config['loss']}, try other options: 'bpr'"
                " or 'bce'"
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
