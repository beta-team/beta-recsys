"""
Todo: weight decay part 
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import GPUtil
import random
import string

from tqdm import tqdm
from datetime import datetime
from utils import logger, data_util
from utils.monitor import Monitor
from utils.constants import *
from utils.triple_sampler import Sampler
from utils.evaluation import save_result
from datasets import dataset

import torch
from torch.utils.data import DataLoader


def print_dict(dic):
    print("-" * 80)
    print(
        "Configs: \n"
        + "\n".join([str(k) + ":\t" + str(v) for k, v in dic.items()])
        + "\n"
    )
    print("-" * 80)


class TrainEngine(object):
    """
    Initialize configs
    Input with config dic, which is from the command line parameters
    """

    def __init__(self, config):
        # obtain abspath for the project
        if "root_dir" not in config:
            UTILS_ROOT = os.path.dirname(os.path.abspath(__file__))
            config["root_dir"] = os.path.abspath(os.path.join(UTILS_ROOT, ".."))
        # load config file from json
        with open(config["config_file"]) as config_params:
            print("loading config file", config["config_file"])
            json_config = json.load(config_params)
        json_config.update(config)
        config = json_config
        # construct unique model run id, which consist of model name, config id and a timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = "".join(
            [random.choice(string.ascii_letters + string.digits) for n in range(4)]
        )
        config["model_run_id"] = (
            config["model"]
            + "_"
            + config["config_id"]
            + "_"
            + timestamp_str
            + "_"
            + random_str
        )

        # log file
        config["log_file"] = os.path.join(
            config["root_dir"], config["log_dir"], config["model_run_id"]
        )

        """
        Logging
        """
        logging = logger.init_std_logger(config["log_file"])

        """
        file paths to be saved
        """
        config["run_dir"] = os.path.join(
            config["root_dir"], config["run_dir"], config["model_run_id"]
        )
        print("The intermediate statuses will report in folder:", config["run_dir"])

        config["model_ckp_file"] = os.path.join(
            config["root_dir"],
            config["checkpoint_dir"],
            config["model_run_id"] + ".model",
        )
        print("Model checkpoint will save in file:", config["model_ckp_file"])

        config["result_file"] = os.path.join(
            config["root_dir"], config["result_dir"], config["result_file"]
        )
        print("Performance result will save in file:", config["result_file"])
        print("python version:", sys.version)
        print("pytorch version:", torch.__version__)
        print_dict(config)
        self.config = config
        self.gpu_id = self.get_gpu()
        """
        monitoring resources of this application
        """
        self.build_dataset()

    """
     Get a gpu id list sorted by the most available memory
    """

    def get_gpu(self):
        DEVICE_ID_LIST = GPUtil.getAvailable(
            order="memory", limit=3
        )  # get the fist gpu with the lowest load
        if len(DEVICE_ID_LIST) < 1:
            gpu_id = None
            device_str = "cpu"
        else:
            gpu_id = DEVICE_ID_LIST[0]
            # need to set 0 if ray only specify 1 gpu
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                if len(os.environ["CUDA_VISIBLE_DEVICES"].split()) == 1:
                    #                     gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
                    gpu_id = 0
                    print("Find only one gpu with id: ", gpu_id)
                    device_str = "cuda:" + str(gpu_id)
            #                     print(os.system("nvidia-smi"))
            else:
                print("Get a gpu id list sorted by the most available memory:", gpu_id)
                device_str = "cuda:" + str(gpu_id)
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        self.config["device_str"] = device_str
        return gpu_id

    def sample_triple(self):
        """
        Sample triples or load triples samples from files
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
        my_sampler = Sampler(self.data.train, sample_file, self.config["n_sample"],)
        if self.config["temp_train"] and self.config["data_split"] == "temporal":
            self.train_data = my_sampler.sample_by_time(self.config["temp_train"])
        else:
            self.train_data = my_sampler.sample()

    def build_dataset(self):
        self.data = data_util.Dataset(self.config)
        self.train_data = self.data.train
        self.config["n_users"] = self.data.n_users
        self.config["n_items"] = self.data.n_items

    """
    Default data builder
    """

    def build_data_loader(self):
        return DataLoader(
            torch.LongTensor(self.train_data.to_numpy(), device=self.engine.device),
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
        )

    def build_temporal_data_loader(self, t):
        t_triple_df = self.train_data[self.train_data["T"] == t]
        data_loader = DataLoader(
            torch.LongTensor(t_triple_df.to_numpy()),
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        return data_loader

    """
    Default train
    """

    def train(self):
        assert hasattr(self, "engine"), "Please specify the exact model engine !"
        monitor = Monitor(log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id)
        """
        init model
        """
        print("init model ...")
        self.engine.data = self.data
        print("strat traning... ")
        best_performance = 0
        n_no_update = 0
        epoch_bar = tqdm(range(self.config["num_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print("Epoch {} starts !".format(epoch))
            print("-" * 80)
            data_loader = self.build_data_loader()
            self.engine.train_an_epoch(data_loader, epoch_id=epoch)
            result = self.engine.evaluate(self.data.validate[0], epoch_id=epoch)
            test_result = self.engine.evaluate(self.data.test[0], epoch_id=epoch)
            self.engine.record_performance(result, test_result, epoch_id=epoch)
            if result[self.config["validate_metric"]] > best_performance:
                n_no_update = 0
                print_dict(result)
                self.engine.save_checkpoint(model_dir=self.config["model_ckp_file"])
                best_performance = result[self.config["validate_metric"]]
            else:
                n_no_update += 1

            if n_no_update >= MAX_N_UPDATE:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        MAX_N_UPDATE
                    )
                )
                break
            """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
            lr = self.config["lr"] * (0.5 ** (epoch // 10))
            for param_group in self.engine.optimizer.param_groups:
                param_group["lr"] = lr
        self.config["run_time"] = monitor.stop()
        return best_performance

    def test(self):
        """
        Prediction and evalution on test set
        """
        result_para = {
            "model": [self.config["model"]],
            "dataset": [self.config["dataset"]],
            "data_split": [self.config["data_split"]],
            "temp_train": [self.config["temp_train"]],
            "emb_dim": [int(self.config["emb_dim"])],
            "lr": [self.config["lr"]],
            "batch_size": [int(self.config["batch_size"])],
            "optimizer": [self.config["optimizer"]],
            "num_epoch": [self.config["num_epoch"]],
            "model_run_id": [self.config["model_run_id"]],
            "run_time": [self.config["run_time"]],
        }

        """
        load the best model in terms of the validate
        """
        self.engine.resume_checkpoint(model_dir=self.config["model_ckp_file"])
        for i in range(10):
            print("Testing test_data", i, "...")
            result = self.engine.evaluate(self.data.test[i], epoch_id=0)
            print_dict(result)
            result.update(result_para)
            result_df = pd.DataFrame(result)
            save_result(result_df, self.config["result_file"])