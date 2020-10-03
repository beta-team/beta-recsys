"""isort:skip_file."""
import argparse
import json
import os
import sys

sys.path.append("../")

import numpy as np

from beta_rec.core.train_engine import TrainEngine
from beta_rec.models.cmn import cmnEngine
from beta_rec.models.pairwise_gmf import PairwiseGMFEngine
from beta_rec.utils.common_util import ensureDir, update_args
from beta_rec.utils.constants import MAX_N_UPDATE
from beta_rec.utils.monitor import Monitor


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run cmn..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/cmn_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")

    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()


class cmn_train(TrainEngine):
    """An instance class from the TrainEngine base class."""

    def __init__(self, config):
        """Initialize CMN_train Class.

        Args:
            config (dict): All the parameters for the model.
        """
        self.config = config
        super(cmn_train, self).__init__(self.config)
        self.load_dataset()
        self.gmfengine = PairwiseGMFEngine(self.config)
        self.cmnengine = cmnEngine
        self.gpu_id, self.config["device_str"] = self.get_device()

    def train_gmf(self):
        """Train GMF."""
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.model_dir = os.path.join(
            self.config["model_save_dir"], self.config["save_name"]
        )
        for epoch in range(config["max_epoch"]):
            print(f"Epoch {epoch} starts !")
            print("-" * 80)
            if epoch > 0 and self.eval_engine.n_no_update == 0:
                # previous epoch have already obtained better result
                self.gmfengine.save_checkpoint(model_dir=self.model_dir)

            if self.eval_engine.n_no_update >= MAX_N_UPDATE:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        MAX_N_UPDATE
                    )
                )
                break

            train_loader = self.data
            self.gmfengine.train_an_epoch(epoch_id=epoch, train_loader=train_loader)

        print("Saving embeddings to: %s" % self.config["model_save_dir"])
        user_embed, item_embed, v = (
            self.gmfengine.model.user_memory.weight.detach().cpu(),
            self.gmfengine.model.item_memory.weight.detach().cpu(),
            self.gmfengine.model.v.weight.detach().cpu(),
        )
        embed_dir = os.path.join(self.config["model_save_dir"], "pretain/embeddings")
        ensureDir(embed_dir)
        np.savez(embed_dir, user=user_embed, item=item_embed, v=v)
        self.config["run_time"] = self.monitor.stop()

        return np.array(user_embed), np.array(item_embed)

    def train(self):
        """Train the model."""
        if self.config["pretrain"] == "gmf":
            user_embed, item_embed = self.train_gmf()
            model = self.cmnengine(
                self.config, user_embed, item_embed, self.data.item_users_list
            )
            self.monitor = Monitor(
                log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
            )
            self.model_dir = os.path.join(
                self.config["model_save_dir"], self.config["save_name"]
            )
            for epoch in range(config["max_epoch"]):
                print(f"Epoch {epoch} starts !")
                print("-" * 80)
                if epoch > 0 and self.eval_engine.n_no_update == 0:
                    # previous epoch have already obtained better result
                    model.save_checkpoint(model_dir=self.model_dir)

                if self.eval_engine.n_no_update >= MAX_N_UPDATE:
                    print(
                        "Early stop criterion triggered, no performance update for {:} times".format(
                            MAX_N_UPDATE
                        )
                    )
                    break

                train_loader = self.data
                model.train_an_epoch(epoch_id=epoch, train_loader=train_loader)

                self.eval_engine.train_eval(
                    self.data.valid[0], self.data.test[0], model.model, epoch
                )
            self.config["run_time"] = self.monitor.stop()
            self.eval_engine.test_eval(self.data.test, model.model)


if __name__ == "__main__":
    args = parse_args()
    config_file = args.config_file
    with open(config_file) as config_params:
        config = json.load(config_params)
    update_args(config, args)
    cmn = cmn_train(config)
    cmn.train()
