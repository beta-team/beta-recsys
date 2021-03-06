"""isort:skip_file."""
import argparse
import sys
from datetime import datetime

sys.path.append("../")

import cornac
import numpy as np
import pandas as pd

import beta_rec.utils.constants as Constants
import beta_rec.utils.evaluation as eval_model
from beta_rec.data import deprecated_data
from beta_rec.utils import logger
from beta_rec.utils.common_util import save_to_csv


base_string = "abcdefghijklmnopqrstuvwxyz"

config = {
    "dataset": "ml_100k",
    "data_split": "temporal",
    "result_file": "cornac_result.csv",
    "root_dir": "../",
}


def parse_args():
    """Parse args from command line.

    Returns:
        args object.
    """
    parser = argparse.ArgumentParser(description="Run cornac model..")
    # If the following settings are specified with command line,
    # these settings will be updated.
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
        "--test_percent",
        nargs="?",
        type=float,
        help="Options are: leave_one_out and temporal",
    )
    parser.add_argument(
        "--root_dir",
        nargs="?",
        type=str,
        help="working directory",
    )
    parser.add_argument(
        "--toy",
        nargs="?",
        type=int,
        help="working directory",
    )
    return parser.parse_args()


"""
update hyperparameters from command line
"""


def update_args(config, args):
    """Update args."""
    #     print(vars(args))
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
            print("Received parameters form comand line:", k, v)


def my_eval(eval_data_df, model):
    """Missing Doc."""
    u_indices = eval_data_df[Constants.DEFAULT_USER_COL].to_numpy()
    i_indices = eval_data_df[Constants.DEFAULT_ITEM_COL].to_numpy()
    r_preds = np.fromiter(
        (
            model.score(user_idx, item_idx).item()
            for user_idx, item_idx in zip(u_indices, i_indices)
        ),
        dtype=np.float,
        count=len(u_indices),
    )

    pred_df = pd.DataFrame(
        {
            Constants.DEFAULT_USER_COL: u_indices,
            Constants.DEFAULT_ITEM_COL: i_indices,
            Constants.DEFAULT_PREDICTION_COL: r_preds,
        }
    )

    result_dic = {}
    TOP_K = [5, 10, 20]
    if type(TOP_K) != list:
        TOP_K = [TOP_K]
    if 10 not in TOP_K:
        TOP_K.append(10)
    metrics = ["ndcg_at_k", "precision_at_k", "recall_at_k", "map_at_k"]

    for k in TOP_K:
        for metric in metrics:
            eval_metric = getattr(eval_model, metric)
            result = eval_metric(eval_data_df, pred_df, k=k)
            result_dic[metric + "@" + str(k)] = result
    result_dic.update(config)
    result_df = pd.DataFrame(result_dic, index=[0])
    save_to_csv(result_df, config["result_file"])


if __name__ == "__main__":
    args = parse_args()
    update_args(config, args)
    root_dir = config["root_dir"]

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (
        root_dir
        + "logs/cornac"
        + "_"
        + config["dataset"]
        + "_"
        + config["data_split"]
        + time_str
    )
    config["result_file"] = (
        root_dir
        + "results/cornac"
        + "_"
        + config["dataset"]
        + "_"
        + config["data_split"]
        + ".csv"
    )
    """
    init logger
    """
    logger.init_std_logger(log_file)

    #     cornac.eval_methods.base_method.rating_eval = rating_eval
    # Load the built-in MovieLens 100K dataset (will be downloaded if not cached):

    # Here we are comparing Biased MF, PMF, and BPR:
    pop = cornac.models.most_pop.recom_most_pop.MostPop(name="MostPop")

    mf = cornac.models.MF(
        k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123
    )
    pmf = cornac.models.PMF(
        k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123
    )
    bpr = cornac.models.BPR(
        k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123
    )

    vaecf = cornac.models.vaecf.recom_vaecf.VAECF(
        name="VAECF",
        k=10,
        autoencoder_structure=[20],
        act_fn="tanh",
        likelihood="mult",
        n_epochs=100,
        batch_size=100,
        learning_rate=0.001,
        beta=1.0,
        trainable=True,
        verbose=False,
        seed=None,
        use_gpu=True,
    )

    nmf = cornac.models.NMF(
        k=15,
        max_iter=50,
        learning_rate=0.005,
        lambda_u=0.06,
        lambda_v=0.06,
        lambda_bu=0.02,
        lambda_bi=0.02,
        use_bias=False,
        verbose=True,
        seed=123,
    )

    neumf = cornac.models.ncf.recom_neumf.NeuMF(
        name="NeuMF",
        num_factors=8,
        layers=(64, 32, 16, 8),
        act_fn="relu",
        reg_mf=0.0,
        reg_layers=(0.0, 0.0, 0.0, 0.0),
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
    )

    models = [pop, mf, pmf, bpr, vaecf, nmf, neumf]
    # add our own eval
    data = deprecated_data.GroceryData(config)

    num_users = data.n_users
    num_items = data.n_items
    uid_map = data.user2id
    iid_map = data.item2id

    train_uir_tuple = [
        data.train["col_user"].to_numpy(),
        data.train["col_item"].to_numpy(),
        data.train["col_rating"].to_numpy(),
    ]

    train_data = cornac.dataset.GroceryData(
        num_users,
        num_items,
        uid_map,
        iid_map,
        train_uir_tuple,
        timestamps=None,
        seed=None,
    )

    test_df_li = data.test

    for model in models:
        config["model"] = str(model.__class__).split(".")[-1].replace(">", "").strip(
            "'\""
        ) + datetime.now().strftime("_%Y%m%d_%H%M%S")
        model.fit(train_data)
        #         for test_data in test_df_li:
        #             my_eval(test_data, model)
        my_eval(test_df_li[0], model)
