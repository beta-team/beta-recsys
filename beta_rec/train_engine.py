import sys
import os
import json
import GPUtil
import string
import random
import torch
from tqdm import tqdm
from datetime import datetime
from beta_rec.eval_engine import EvalEngine
from beta_rec.utils import logger, data_util
from beta_rec.utils.monitor import Monitor
from beta_rec.utils.constants import MAX_N_UPDATE
from beta_rec.utils.common_util import set_seed, initialize_folders, print_dict_as_table
from beta_rec.utils.triple_sampler import Sampler
from torch.utils.data import DataLoader


def prepare_env(config):
    """Prepare running environment
        - Load parameters from json files.
        - Initialize system folders, model name and the paths to be saved.
        - Initialize resource monitor.
        - Initialize random seed.
        - Initialize logging.

    Args:
        config (dict): Global configs.

    """
    # obtain abspath for the project
    # You need specified it if it is running in the container.
    if "root_dir" not in config:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        config["root_dir"] = os.path.abspath(os.path.join(file_dir, ".."))

    # load config file from json
    with open(config["config_file"]) as config_params:
        print("loading config file", config["config_file"])
        json_config = json.load(config_params)

    # update global parameters with these parameters received from the command line .
    json_config.update(config)
    config = json_config

    # construct unique model run id, which consist of model name, config id and a timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join([random.choice(string.ascii_lowercase) for n in range(6)])
    config["model_run_id"] = (
        config["model"]
        + "_"
        + config["config_id"]
        + "_"
        + timestamp_str
        + "_"
        + random_str
    )
    set_seed(config["seed"] if "seed" in config else 2020)
    initialize_folders(config["root_dir"])

    # Initialize log file
    config["log_file"] = os.path.join(
        config["root_dir"], config["log_dir"], config["model_run_id"]
    )
    logger.init_std_logger(config["log_file"])

    print("python version:", sys.version)
    print("pytorch version:", torch.__version__)

    #  File paths to be saved
    config["run_dir"] = os.path.join(
        config["root_dir"], config["run_dir"], config["model_run_id"]
    )
    print(
        "The intermediate running statuses will be reported in folder:",
        config["run_dir"],
    )

    #  Model checkpoints paths to be saved
    config["model_save_dir"] = os.path.join(
        config["root_dir"], config["checkpoint_dir"], config["model_run_id"]
    )
    os.mkdir(config["model_save_dir"])
    print("Model checkpoint will save in file:", config["model_save_dir"])

    config["result_file"] = os.path.join(
        config["root_dir"], config["result_dir"], config["result_file"]
    )
    print("Performance result will save in file:", config["result_file"])

    # remove comments

    print_dict_as_table(config, "Model configs")
    return config


def get_device():
    """
        Get one gpu id that have the most available memory.
    Returns:
        (int, str): The gpu id (None if no available gpu) and the the device string (pytorch style).
    """
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
        #                     print(os.system("nvidia-smi"))
        else:
            print("Get a gpu id list sorted by the most available memory:", gpu_id)
            device_str = "cuda:" + str(gpu_id)
    return gpu_id, device_str


class TrainEngine(object):
    """Training engine for all the models.

    """

    def __init__(self, config):
        """Initialing

        Args:
            config (dict): Config dict received from command line. Should have the config["config_file"].

        Attributes:
            dataset (Dataset): A dataset containing DataFrame of train, validation and test.
            train_data (DataLoader): Train DataLoader, need to be implement.
            monitor (Monitor): An monitor object that monitor the computational resources.
            engine (Model Engine)


        """
        self.dataset = None
        self.train_data = None
        self.monitor = None
        self.engine = None
        self.config = prepare_env(config)
        self.gpu_id, self.config["device_str"] = get_device()
        self.eval_engine = EvalEngine(self.config)
        self.build_dataset()

    def sample_triple(self):
        """
        Sample triples or load triples samples from files. Only applicable for basket based recommenders
        Returns:
            None

        """
        # need to be specified if need samples
        self.config["sample_dir"] = os.path.join(
            self.config["root_dir"], self.config["sample_dir"]
        )
        sample_file = (
            self.config["sample_dir"]
            + "triple_"
            + self.config["dataset"]
            + "_"
            + str(self.config["percent"] * 100)
            + "_"
            + str(self.config["n_sample"])
            + "_"
            + str(self.config["temp_train"])
            + ".csv"
        )
        my_sampler = Sampler(self.dataset.train, sample_file, self.config["n_sample"])
        self.train_data = my_sampler.sample()

    def build_dataset(self):
        """ Default implementation of building dataset

        Returns:
            None

        """
        self.dataset = data_util.Dataset(self.config)
        self.config["n_users"] = self.dataset.n_users
        self.config["n_items"] = self.dataset.n_items

    # noinspection PyTypeChecker
    def build_data_loader(self):
        """ Default data builder

        Returns:
            DataLoader

        """
        return DataLoader(
            torch.LongTensor(self.train_data.to_numpy()).to(self.engine.device),
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def train(self):
        """Default train implementation

        """
        assert hasattr(self, "engine"), "Please specify the exact model engine !"
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.engine.data = self.dataset
        print("Start training... ")
        best_performance = 0
        epoch_bar = tqdm(range(self.config["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print(f"Epoch {epoch} starts !")
            print("-" * 80)
            if epoch > 0 and self.eval_engine.n_no_update == 0:
                # previous epoch have already obtained better result
                self.engine.save_checkpoint(
                    model_dir=self.config["model_save_dir"] + "model.cpk"
                )

            if self.eval_engine.n_no_update >= MAX_N_UPDATE:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        MAX_N_UPDATE
                    )
                )
                break
            data_loader = self.build_data_loader()
            self.engine.train_an_epoch(data_loader, epoch_id=epoch)
            self.eval_engine.train_eval(
                self.dataset.validate[0], self.dataset.test[0], self.engine.model, epoch
            )
            """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
            lr = self.config["lr"] * (0.5 ** (epoch // 10))
            for param_group in self.engine.optimizer.param_groups:
                param_group["lr"] = lr
        self.config["run_time"] = self.monitor.stop()
        return best_performance

    def test(self):
        """Evaluate the performance for the testing sets based on the final model.

        """
        self.eval_engine.test_eval(self.dataset.test, self.engine.model)
