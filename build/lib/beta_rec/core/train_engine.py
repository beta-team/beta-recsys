import json
import os
import random
import string
import sys
from datetime import datetime

import GPUtil
import torch
from ax.service.ax_client import AxClient
from ray import tune
from ray.tune.suggest.ax import AxSearch
from torch.utils.data import DataLoader
from tqdm import tqdm

from beta_rec.core.eval_engine import EvalEngine
from beta_rec.data.grocery_data import GroceryData
from beta_rec.utils import logger
from beta_rec.utils.common_util import (
    ensureDir,
    initialize_folders,
    print_dict_as_table,
    set_seed,
    update_args,
)
from beta_rec.utils.constants import MAX_N_UPDATE
from beta_rec.utils.monitor import Monitor


def prepare_env(args):
    """Prepare running environment
        - Load parameters from json files.
        - Initialize system folders, model name and the paths to be saved.
        - Initialize resource monitor.
        - Initialize random seed.
        - Initialize logging.

    Args:
        args (ArgumentParser): Args received from command line. Should have the parameter of args.config_file.

    """

    # Load config file from json
    with open(args.config_file) as config_params:
        print(f"loading config file {args.config_file}")
        config = json.load(config_params)

    # Update configs based on the received args from the command line .
    update_args(config, args)

    # obtain abspath for the project
    if config["system"]["root_dir"] == "default":
        file_dir = os.path.dirname(os.path.abspath(__file__))
        config["system"]["root_dir"] = os.path.abspath(
            os.path.join(file_dir, "..", "..")
        )

    # construct unique model run id, which consist of model name, config id and a timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join([random.choice(string.ascii_lowercase) for n in range(6)])
    config["system"]["model_run_id"] = (
        config["model"]["model"]
        + "_"
        + config["model"]["config_id"]
        + "_"
        + timestamp_str
        + "_"
        + random_str
    )

    # Initialize random seeds
    set_seed(config["system"]["seed"] if "seed" in config["system"] else 2020)

    # Initialize working folders
    initialize_folders(config["system"]["root_dir"])

    config["system"]["process_dir"] = os.path.join(
        config["system"]["root_dir"], config["system"]["process_dir"]
    )

    # Initialize log file
    config["system"]["log_file"] = os.path.join(
        config["system"]["root_dir"],
        config["system"]["log_dir"],
        config["system"]["model_run_id"],
    )
    logger.init_std_logger(config["system"]["log_file"])

    print("Python version:", sys.version)
    print("pytorch version:", torch.__version__)

    #  File paths to be saved
    config["model"]["run_dir"] = os.path.join(
        config["system"]["root_dir"],
        config["system"]["run_dir"],
        config["system"]["model_run_id"],
    )
    config["system"]["run_dir"] = config["model"]["run_dir"]
    print(
        "The intermediate running statuses will be reported in folder:",
        config["system"]["run_dir"],
    )

    #  Model checkpoints paths to be saved
    config["system"]["model_save_dir"] = os.path.join(
        config["system"]["root_dir"],
        config["system"]["checkpoint_dir"],
        config["system"]["model_run_id"],
    )
    ensureDir(config["system"]["model_save_dir"])
    print("Model checkpoint will save in file:", config["system"]["model_save_dir"])

    config["system"]["result_file"] = os.path.join(
        config["system"]["root_dir"],
        config["system"]["result_dir"],
        config["system"]["result_file"],
    )
    print("Performance result will save in file:", config["system"]["result_file"])

    print_dict_as_table(config["system"], "System configs")
    return config


class TrainEngine(object):
    """Training engine for all the models.
    # You need specified root_dir from CLS if it is to run in the container.
        Args:
            args (ArgumentParser): Args received from command line. Should have the parameter of args.config_file.

        Attributes:
            data (GroceryData): A dataset containing DataFrame of train, validation and test.
            train_data (DataLoader): Extracted training data or train DataLoader, need to be implement.
            monitor (Monitor): An monitor object that monitor the computational resources.
            engine (Model Engine)
    """

    def __init__(self, args):
        self.data = None
        self.train_loader = None
        self.monitor = None
        self.engine = None
        self.config = prepare_env(args)
        self.gpu_id, self.config["model"]["device_str"] = self.get_device()
        self.eval_engine = EvalEngine(self.config)

    def get_device(self):
        """
            Get one gpu id that have the most available memory.
        Returns:
            (int, str): The gpu id (None if no available gpu) and the the device string (pytorch style).
        """
        if "device" in self.config["system"]:
            if self.config["system"]["device"] == "cpu":
                return (None, "cpu")
            elif (
                "cuda" in self.config["system"]["device"]
            ):  # receive an string with "cuda:#"
                return (
                    int(self.config["system"]["device"].replace("cuda", "")),
                    self.config["system"]["device"],
                )
            elif len(self.config["system"]["device"]) < 1:  # receive an int string
                return (
                    int(self.config["system"]["device"]),
                    "cuda:" + self.config["system"]["device"],
                )

        gpu_id_list = GPUtil.getAvailable(
            order="memory", limit=3
        )  # get the fist gpu with the lowest load
        if len(gpu_id_list) < 1:
            gpu_id = None
            device_str = "cpu"
        else:
            gpu_id = gpu_id_list[0]
            # need to set 0 if ray only specify 1 gpu
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                if len(os.environ["CUDA_VISIBLE_DEVICES"].split()) == 1:
                    #  gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
                    gpu_id = 0
                    print("Find only one gpu with id: ", gpu_id)
                    device_str = "cuda:" + str(gpu_id)
            # print(os.system("nvidia-smi"))
            else:
                print("Get a gpu with the most available memory :", gpu_id)
                device_str = "cuda:" + str(gpu_id)
        return gpu_id, device_str

    def load_dataset(self):
        """ Default implementation of building Data
        """
        self.data = GroceryData(self.config)
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items

    # noinspection PyTypeChecker
    def build_data_loader(self):
        """ Default implementation of building DataLoader
        """
        self.train_loader = DataLoader(
            torch.LongTensor(self.data.sample_triple()).to(self.engine.device),
            batch_size=self.config["model"]["batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def check_early_stop(self, engine, model_dir, epoch):
        """ Check if early stop criterion is triggered
        Save model if previous epoch have already obtained better result

        Args:
            epoch (int): epoch num

        Returns:
            bool: True: if early stop criterion is triggered,  False: else
        """
        if epoch > 0 and self.eval_engine.n_no_update == 0:
            # save model if previous epoch have already obtained better result
            engine.save_checkpoint(model_dir=model_dir)

        if self.eval_engine.n_no_update >= MAX_N_UPDATE:
            # stop training if early stop criterion is triggered
            print(
                "Early stop criterion triggered, no performance update for {:} times".format(
                    MAX_N_UPDATE
                )
            )
            return True
        return False

    def _train(self, engine, train_loader, save_dir):
        self.eval_engine.flush()
        epoch_bar = tqdm(range(self.config["model"]["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            if self.config["dataset"]["validate"]:
                self.eval_engine.train_eval(
                    self.data.valid[0], self.data.test[0], engine.model, epoch
                )
            else:
                self.eval_engine.train_eval(
                    None, self.data.test[0], engine.model, epoch
                )

    def tune(self, runable):
        """
        Tune parameters unsing ray.tune and ax
        Returns:

        """
        ax = AxClient(enforce_sequential_optimization=False)
        # verbose_logging=False,
        ax.create_experiment(
            name=self.config["model"]["model"],
            parameters=self.config["tunable"],
            objective_name="valid_metric",
        )
        tune.run(
            runable,
            num_samples=30,
            search_alg=AxSearch(ax),  # Note that the argument here is the `AxClient`.
            verbose=2,  # Set this level to 1 to see status updates and to 2 to also see trial results.
            # To use GPU, specify: resources_per_trial={"gpu": 1}.
        )

        # analysis = tune.run(
        #     runable, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

    def test(self):
        """Evaluate the performance for the testing sets based on the final model.

        """
        self.eval_engine.test_eval(self.data.test, self.engine.model)
