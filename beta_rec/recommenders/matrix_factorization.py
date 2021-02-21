import os
import time

from munch import munchify
from ray import tune

from ..core.recommender import Recommender
from ..models.mf import MFEngine
from ..utils.monitor import Monitor


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
    tune.report(
        valid_metric=result["valid_metric"],
        model_save_dir=result["model_save_dir"],
    )


class MatrixFactorization(Recommender):
    """The Matrix Factorization Model."""

    def __init__(self, config):
        """Initialize the config of this recommender.

        Args:
            config:
        """
        super(MatrixFactorization, self).__init__(config, name="MF")

    def init_engine(self, data):
        """Initialize the required parameters for the model.

        Args:
            data: the Dataset object.

        """
        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = MFEngine(self.config)

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
