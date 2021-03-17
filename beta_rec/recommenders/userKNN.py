import os
import time

from munch import munchify
from ray import tune

from ..core.recommender import Recommender
from ..models.userKNN import UserKNNEngine
from ..utils.monitor import Monitor


def tune_train(config):
    """Train the model with a hyper-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    data = config["data"]
    train_engine = UserKNN(munchify(config))
    result = train_engine.train(data)
    while train_engine.eval_engine.n_worker > 0:
        time.sleep(20)
    tune.report(
        valid_metric=result["valid_metric"],
        model_save_dir=result["model_save_dir"],
    )


class UserKNN(Recommender):
    """The User-based K Nearest Neighbour Model."""

    def __init__(self, config):
        """Initialize the config of this recommender.

        Args:
            config:
        """
        super(UserKNN, self).__init__(config, name="UserKNN")

    def init_engine(self, data):
        """Initialize the required parameters for the model.

        Args:
            data: the Dataset object.

        """
        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.engine = UserKNNEngine(self.config)

    def train(self, data):
        """Training the model.

        Args:
            data: the Dataset object.

        Returns:
            dict: {}

        """
        self.gpu_id, self.config["device_str"] = self.get_device()  # Train the model.

        self.config["model"]["n_users"] = data.n_users
        self.config["model"]["n_items"] = data.n_items
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.init_engine(data)
        print(type(data.train))
        print(data.train.head())
        self.engine.model.prepare_model(data)
        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self.config["run_time"] = self.monitor.stop()
        return "data loaded"
