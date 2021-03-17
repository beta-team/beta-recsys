"""isort:skip_file."""
import argparse
import os
import sys
import time

sys.path.append("../")

from ray import tune

from beta_rec.core.train_engine import TrainEngine
from beta_rec.data.deprecated_data_base import DataLoaderBase
from beta_rec.models.gmf import GMFEngine
from beta_rec.models.mlp import MLPEngine
from beta_rec.models.ncf import NeuMFEngine
from beta_rec.utils.common_util import DictToObject, str2bool
from beta_rec.utils.monitor import Monitor


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run NCF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/ncf_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        help="Options are: 'mlp', 'gmf', 'ncf_end', and 'ncf_pre'",
    )
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
        help="Working directory",
    )
    parser.add_argument(
        "--tune",
        nargs="?",
        type=str2bool,
        help="Tune parameter",
    )
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initial learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


class NCF_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config):
        """Initialize NCF_train Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(NCF_train, self).__init__(self.config)
        self.load_dataset()
        self.build_data_loader()
        self.gpu_id, self.config["model"]["device_str"] = self.get_device()

    def build_data_loader(self):
        """Missing Doc."""
        # ToDo: Please define the directory to store the adjacent matrix
        self.sample_generator = DataLoaderBase(ratings=self.data.train)
        self.config["model"]["num_batch"] = (
            self.data.n_train // self.config["model"]["batch_size"] + 1
        )
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items

    def train(self):
        """Train the model."""
        # Options are: 'mlp', 'gmf', 'ncf_end', and 'ncf_pre';
        # Train NeuMF without pre-train
        if self.config["model"]["model"] == "ncf_end":
            self.train_ncf()
        elif self.config["model"]["model"] == "gmf":
            self.train_gmf()
        elif self.config["model"]["model"] == "mlp":
            self.train_mlp()
        elif self.config["model"]["model"] == "ncf_pre":
            self.train_gmf()
            self.train_mlp()
            self.train_ncf()
        else:
            raise ValueError(
                "Model type error: Options are: 'mlp', 'gmf', 'ncf_end', and 'ncf_pre'."
            )

    def train_ncf(self):
        """Train NeuMF."""
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        train_loader = self.sample_generator.instance_a_train_loader(
            self.config["model"]["num_negative"], self.config["model"]["batch_size"]
        )
        self.engine = NeuMFEngine(self.config)
        self.neumf_save_dir = os.path.join(
            self.config["system"]["model_save_dir"],
            self.config["model"]["neumf_config"]["save_name"],
        )
        self._train(self.engine, train_loader, self.neumf_save_dir)
        self.config["run_time"] = self.monitor.stop()
        self.eval_engine.test_eval(self.data.test, self.engine.model)

    def train_gmf(self):
        """Train GMF."""
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        train_loader = self.sample_generator.instance_a_train_loader(
            self.config["model"]["num_negative"], self.config["model"]["batch_size"]
        )
        # Train GMF
        self.engine = GMFEngine(self.config)
        self.gmf_save_dir = os.path.join(
            self.config["system"]["model_save_dir"],
            self.config["model"]["gmf_config"]["save_name"],
        )
        self._train(self.engine, train_loader, self.gmf_save_dir)
        while self.eval_engine.n_worker:
            print("Wait 15s for the complete of eval_engine.n_worker")
            time.sleep(15)  # wait the
        self.config["run_time"] = self.monitor.stop()
        self.eval_engine.test_eval(self.data.test, self.engine.model)

    def train_mlp(self):
        """Train MLP."""
        # Train MLP
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        train_loader = self.sample_generator.instance_a_train_loader(
            self.config["model"]["num_negative"], self.config["model"]["batch_size"]
        )
        self.engine = MLPEngine(self.config)
        self.mlp_save_dir = os.path.join(
            self.config["system"]["model_save_dir"],
            self.config["model"]["mlp_config"]["save_name"],
        )
        self._train(self.engine, train_loader, self.mlp_save_dir)

        while self.eval_engine.n_worker:
            print("Wait 15s for the complete of eval_engine.n_worker")
            time.sleep(15)  # wait the
        self.config["run_time"] = self.monitor.stop()
        self.eval_engine.test_eval(self.data.test, self.engine.model)


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray).

    Args:
        config (dict): All the parameters for the model.
    """
    train_engine = NCF_train(DictToObject(config))
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
        train_engine = NCF_train(args)
        train_engine.tune(tune_train)
    else:
        print("Run application with single config ...")
        train_engine = NCF_train(args)
        train_engine.train()
        train_engine.test()
