import sys
from beta_rec.datasets.dunnhumby import Dunnhumby
sys.path.append("../")
import os
import json
import numpy as np
import argparse
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from beta_rec.models.NGCF import NGCF
from beta_rec.models.NGCF import NGCFEngine
from beta_rec.datasets.NGCF_data_utils import Data
from beta_rec.datasets import dataset
from beta_rec.datasets.movielens import Movielens_100k


def parse_args():
    """
    Parse args from command line
    Returns:

    """
    parser = argparse.ArgumentParser(description="Run NGCF..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/NGCF_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )

    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Initialize learning rate.")
    parser.add_argument("--num_epoch", nargs="?", type=int, help="Number of max epoch.")

    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    return parser.parse_args()

def update_args(config, args):
    """Update config parameters by the received parameters from command line

        Args:
            config (dict): Initial dict of the parameters from JOSN config file.
            args (object): An argparse Argument object with attributes being the parameters to be updated.

        Returns:
            None
    """
    for k, v in vars(args).items():
        if v != None:
            config[k] = v
            print("Received parameters form comand line:", k, v)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    config_file = args.config_file
    with open(config_file) as config_params:
        config = json.load(config_params)

    dataset = Movielens_100k()
    dataset.preprocess()
    train, vad, test = dataset.load_leave_one_out(n_test=1)
    # ToDo: Please define the directory to store the adjacent matrix
<<<<<<< HEAD
    data_loader = Data(path=dataset.dataset_dir,train=train, test=test[0], vad=vad[0],batch_size=int(config["batch_size"]))
    plain_adj, norm_adj, mean_adj = data_loader.get_adj_mat()
    norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
    vad = data_loader.vad
    test = data_loader.test
=======
    data_loader = Data(path=dataset.dataset_dir,df=train, batch_size=int(config["batch_size"]))
    plain_adj, norm_adj, mean_adj = data_loader.get_adj_mat()
    norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
>>>>>>> 78c8c26c9cefb1481d12f9b2ca98d26ed37ed5fc

    config["norm_adj"] = norm_adj
    config["num_batch"] = data_loader.n_train // config["batch_size"] + 1
    config["num_users"] = data_loader.n_users
    config["num_items"] = data_loader.n_items
    engine = NGCFEngine(config)

    counter = 0
    for epoch in range(config["num_epoch"]):
        users, pos_items, neg_items = data_loader.sample()
        engine.train_an_epoch(epoch_id=epoch, user=users,pos_i=pos_items,neg_i=neg_items)
    # ToDo: Please enable early stop, log, time monitor
        counter+=1
<<<<<<< HEAD
        if counter % 1 == 0:
            result = engine.evaluate(eval_data_df=vad,epoch_id=epoch)
=======
        if counter % 10 == 0:
            result = engine.evaluate(eval_data_df=vad[0],epoch_id=epoch)
>>>>>>> 78c8c26c9cefb1481d12f9b2ca98d26ed37ed5fc
            print(result)
    # ToDo: Please save model and result file





