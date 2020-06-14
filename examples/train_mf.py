import argparse
import os

from ray import tune

from beta_rec.core.train_engine import TrainEngine
from beta_rec.data.data_base import DataLoaderBase
from beta_rec.models.mf import MFEngine
from beta_rec.utils.common_util import DictToObject
from beta_rec.utils.monitor import Monitor


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run MF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/mf_default.json",
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
        "--root_dir", nargs="?", type=str, help="Working directory",
    )
    parser.add_argument(
        "--tune", nargs="?", type=bool, help="Tun parameter",
    )
    parser.add_argument(
        "--device", nargs="?", type=str, help="Device",
    )
    parser.add_argument(
        "--loss", nargs="?", type=str, help="loss: bpr or bce",
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


class MF_train(TrainEngine):
    def __init__(self, args):
        print(args)
        super(MF_train, self).__init__(args)

    def build_data_loader(self):
        # ToDo: Please define the directory to store the adjacent matrix
        self.sample_generator = DataLoaderBase(ratings=self.data.train)

    def train(self):
        self.load_dataset()
        self.build_data_loader()
        self.gpu_id, self.config["device_str"] = self.get_device()
        """ Main training navigator

        Returns:

        """
        # Train NeuMF without pre-train
        self.monitor = Monitor(
            log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        if self.config["model"]["loss"] == "bpr":
            train_loader = self.sample_generator.pairwise_negative_train_loader(
                self.config["model"]["batch_size"], self.config["model"]["device_str"]
            )
        elif self.config["model"]["loss"] == "bce":
            train_loader = self.sample_generator.uniform_negative_train_loader(
                self.config["model"]["num_negative"],
                self.config["model"]["batch_size"],
                self.config["model"]["device_str"],
            )
        else:
            raise ValueError(
                f"Unsupported loss type {self.config['loss']}, try other options: 'bpr' or 'bce'"
            )

        self.engine = MFEngine(self.config["model"])
        self.model_save_dir = os.path.join(
            self.config["system"]["model_save_dir"], self.config["model"]["save_name"]
        )
        self._train(self.engine, train_loader, self.model_save_dir)
        self.config["run_time"] = self.monitor.stop()
        return self.eval_engine.best_valid_performance

    def tune(self, runable):
        """
        Tune parameters unsing ray.tune and ax
        Returns:

        """
        # ax = AxClient(enforce_sequential_optimization=False)
        # # verbose_logging=False,
        # ax.create_experiment(
        #     name=self.config["model"]["model"],
        #     parameters=self.config["tunable"],
        #     objective_name="valid_metric",
        # )
        # tune.run(
        #     runable,
        #     num_samples=30,
        #     search_alg=AxSearch(ax),  # Note that the argument here is the `AxClient`.
        #     verbose=2,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        #     # To use GPU, specify: resources_per_trial={"gpu": 1}.
        # )

        analysis = tune.run(
            runable,
            config={"lr": tune.grid_search([0.001, 0.01, 0.1])},
            local_dir="../tune_results",
        )


def tune_train(config):
    """Train the model with a hypyer-parameter tuner (ray)

    Args:
        config (dict): All the parameters for the model

    Returns:

    """
    train_engine = MF_train(DictToObject(config))
    best_performance = train_engine.train()
    tune.track.log(valid_metric=best_performance)
    train_engine.test()


if __name__ == "__main__":
    args = parse_args()
    train_engine = MF_train(args)
    train_engine.tune(tune_train)
    # train_engine.train()
    # train_engine.test()
