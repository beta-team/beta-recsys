import sys
import os
import json
import numpy as np
import argparse
import torch
from beta_rec.train_engine import TrainEngine
from beta_rec.models.ngcf import NGCFEngine
from beta_rec.utils.common_util import update_args
from beta_rec.utils.constants import MAX_N_UPDATE
from beta_rec.utils.monitor import Monitor

sys.path.append("../")


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run NGCF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/ngcf_default.json",
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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class NGCF_train(TrainEngine):
    """ An instance class from the TrainEngine base class

    """

    def __init__(self, config):
        """Constructor

        Args:
            config (dict): All the parameters for the model
        """

        self.config = config
        super(NGCF_train, self).__init__(self.config)
        self.load_dataset()
        self.build_data_loader()
        self.engine = NGCFEngine(self.config)

    def build_data_loader(self):
        # ToDo: Please define the directory to store the adjacent matrix
        plain_adj, norm_adj, mean_adj = self.dataset.get_adj_mat()
        norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
        self.config["norm_adj"] = norm_adj
        self.config["num_batch"] = self.dataset.n_train // config["batch_size"] + 1
        self.config["n_users"] = self.dataset.n_users
        self.config["n_items"] = self.dataset.n_items

    def train(self):
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
                self.engine.save_checkpoint(model_dir=self.model_dir)

            if self.eval_engine.n_no_update >= MAX_N_UPDATE:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        MAX_N_UPDATE
                    )
                )
                break

            train_loader = self.dataset
            self.engine.train_an_epoch(
                epoch_id=epoch, train_loader=train_loader
            )
            self.eval_engine.train_eval(
                self.dataset.valid[0], self.dataset.test[0], self.engine.model, epoch
            )
        self.config["run_time"] = self.monitor.stop()

    def test(self):
        self.engine.resume_checkpoint(model_dir=self.model_dir)
        super(NGCF_train, self).test()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    config_file = args.config_file
    with open(config_file) as config_params:
        config = json.load(config_params)
    update_args(config, args)
    ngcf = NGCF_train(config)
    ngcf.train()
    ngcf.test()
