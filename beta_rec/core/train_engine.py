import json
import os
import random
import string
import sys
from datetime import datetime

import GPUtil
import ray
import torch
from ray import tune
from tabulate import tabulate
from tqdm import tqdm

from ..core.config import find_config
from ..core.eval_engine import EvalEngine
from ..data.base_data import BaseData
from ..datasets.data_load import load_split_dataset
from ..utils import logger
from ..utils.common_util import ensureDir, print_dict_as_table, set_seed, update_args


class TrainEngine(object):
    """Training engine for all the models."""

    def __init__(self, args):
        """Init TrainEngine Class."""
        self.data = None
        self.train_loader = None
        self.monitor = None
        self.engine = None
        self.args = args
        self.config = self.prepare_env()
        self.gpu_id, self.config["model"]["device_str"] = self.get_device()
        self.eval_engine = EvalEngine(self.config)

    def get_device(self):
        """Get one gpu id that have the most available memory.

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
            elif len(self.config["system"]["device"]) == 1:  # receive an gpu id
                return (
                    int(self.config["system"]["device"]),
                    "cuda:" + self.config["system"]["device"],
                )
        device_str = "cpu"
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

    def prepare_env(self):
        """Prepare running environment.

        * Load parameters from json files.
        * Initialize system folders, model name and the paths to be saved.
        * Initialize resource monitor.
        * Initialize random seed.
        * Initialize logging.
        """
        # Load config file from json
        config_file = find_config(self.args.config_file)
        with open(config_file) as config_params:
            print(f"loading config file {config_file}")
            config = json.load(config_params)

        # Update configs based on the received args from the command line .
        update_args(config, self.args)

        # obtain abspath for the project
        config["system"]["root_dir"] = os.path.abspath(config["system"]["root_dir"])

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
        self.initialize_folders(config)

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

        config["system"]["tune_dir"] = os.path.join(
            config["system"]["root_dir"], config["system"]["tune_dir"]
        )

        def get_user_temp_dir():
            tempdir = os.path.join(config["system"]["root_dir"], "tmp")
            print(f"ray temp dir {tempdir}")
            return tempdir

        ray.utils.get_user_temp_dir = get_user_temp_dir

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

    def initialize_folders(self, config):
        """Initialize the whole directory structure of the project."""
        dirs = [
            "log_dir",
            "result_dir",
            "process_dir",
            "checkpoint_dir",
            "run_dir",
            "tune_dir",
            "dataset_dir",
        ]
        base_dir = config["system"]["root_dir"]
        for directory in dirs:
            path = os.path.join(base_dir, config["system"][directory])
            if not os.path.exists(path):
                os.makedirs(path)

    def load_dataset(self):
        """Load dataset."""
        self.data = BaseData(load_split_dataset(self.config))
        self.config["model"]["n_users"] = self.data.n_users
        self.config["model"]["n_items"] = self.data.n_items

    def check_early_stop(self, engine, model_dir, epoch):
        """Check if early stop criterion is triggered.

        Save model if previous epoch have already obtained better result.

        Args:
            epoch (int): epoch num

        Returns:
            bool: True: if early stop criterion is triggered,  False: else
        """
        max_n_update = self.config["model"]["max_n_update"]
        if epoch > 0 and self.eval_engine.n_no_update == 0:
            # save model if previous epoch have already obtained better result
            engine.save_checkpoint(model_dir=model_dir)

        if self.eval_engine.n_no_update >= max_n_update:
            # stop training if early stop criterion is triggered
            print(
                "Early stop criterion triggered, no performance update for"
                f" {max_n_update} times"
            )
            return True
        return False

    def _train(self, engine, train_loader, save_dir, valid_df=None, test_df=None):
        self.eval_engine.flush()
        epoch_bar = tqdm(range(self.config["model"]["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            if (valid_df is None) & (test_df is None):
                self.eval_engine.train_eval(
                    self.data.valid[0], self.data.test[0], engine.model, epoch
                )
            else:
                self.eval_engine.train_eval(valid_df, test_df, engine.model, epoch)

    def _seq_train(
        self, engine, train_loader, save_dir, train_seq, valid_df=None, test_df=None
    ):
        self.eval_engine.flush()
        epoch_bar = tqdm(range(self.config["model"]["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            if (valid_df is None) & (test_df is None):
                self.eval_engine.seq_train_eval(
                    train_seq,
                    self.data.valid[0],
                    self.data.test[0],
                    engine.model,
                    self.config["model"]["maxlen"],
                    epoch,
                )
            else:
                self.eval_engine.seq_train_eval(
                    train_seq,
                    valid_df,
                    test_df,
                    engine.model,
                    self.config["model"]["maxlen"],
                    epoch,
                )

    def _seq_train_time(
        self, engine, train_loader, save_dir, train_seq, valid_df=None, test_df=None
    ):
        self.eval_engine.flush()
        epoch_bar = tqdm(range(self.config["model"]["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            if self.check_early_stop(engine, save_dir, epoch):
                break
            engine.train_an_epoch(train_loader, epoch_id=epoch)
            """evaluate model on validation and test sets"""
            if (valid_df is None) & (test_df is None):
                self.eval_engine.seq_train_eval_time(
                    train_seq,
                    self.data.valid[0],
                    self.data.test[0],
                    engine.model,
                    self.config["model"]["maxlen"],
                    self.config["model"]["time_span"],
                    epoch,
                )
            else:
                self.eval_engine.seq_train_eval_time(
                    train_seq,
                    valid_df,
                    test_df,
                    engine.model,
                    self.config["model"]["maxlen"],
                    self.config["model"]["time_span"],
                    epoch,
                )

    def tune(self, runable):
        """Tune parameters using ray.tune."""
        config = vars(self.args)
        if "tune" in config:
            config["tune"] = False
        if "root_dir" in config and config["root_dir"]:
            config["root_dir"] = os.path.abspath(config["root_dir"])
        else:
            config["root_dir"] = os.path.abspath("..")
        config["config_file"] = os.path.abspath(config["config_file"])
        print(config)
        tunable = self.config["tunable"]
        for para in tunable:
            if para["type"] == "choice":
                config[para["name"]] = tune.grid_search(para["values"])
            if para["type"] == "range":
                values = []
                for val in range(para["bounds"][0], para["bounds"][1] + 1):
                    values.append(val)
                config[para["name"]] = tune.grid_search(values)

        analysis = tune.run(
            runable,
            config=config,
            local_dir=self.config["system"]["tune_dir"],
            # temp_dir=self.config["system"]["tune_dir"] + "/temp",
        )
        df = analysis.dataframe()
        tune_result_dir = os.path.join(
            self.config["system"]["tune_dir"],
            f"{self.config['system']['model_run_id']}_tune_result.csv",
        )
        print(f"Tuning results are saved in {tune_result_dir}")
        df.to_csv(tune_result_dir)
        print(tabulate(df, headers=df.columns, tablefmt="psql"))
        return df

    # def ax_tune(self, runable):
    #     # todo still cannot runable yet.
    #     ax = AxClient(enforce_sequential_optimization=False)
    #     # verbose_logging=False,
    #     ax.create_experiment(
    #         name=self.config["model"]["model"],
    #         parameters=self.config["tunable"],
    #         objective_name="valid_metric",
    #     )
    #     tune.run(
    #         runable,
    #         num_samples=30,
    #         search_alg=AxSearch(ax),  # Note that the argument here is the `AxClient`.
    #         verbose=2,  # Set this level to 1 to see status updates and to 2 to also see trial results.
    #         # To use GPU, specify: resources_per_trial={"gpu": 1}.
    #     )

    def test(self):
        """Evaluate the performance for the testing sets based on the final model."""
        self.eval_engine.test_eval(self.data.test, self.engine.model)
