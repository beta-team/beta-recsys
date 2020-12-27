import argparse
import gzip
import os
import random
import time
import zipfile
from functools import wraps

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tabulate import tabulate

from ..utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_ORDER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)


def normalized_adj_single(adj):
    """Missing docs.

    Args:
        adj:

    Returns:
        None.
    """
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    # norm_adj = adj.dot(d_mat_inv)
    print("generate single-normalized adjacency matrix.")
    return norm_adj.tocoo()


def ensureDir(dir_path):
    """Ensure a dir exist, otherwise create the path.

    Args:
        dir_path (str): the target dir.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def update_args(config, args):
    """Update config parameters by the received parameters from command line.

    Args:
        config (dict): Initial dict of the parameters from JSON config file.
        args (object): An argparse Argument object with attributes being the parameters to be updated.
    """
    args_dic = {}
    for cfg in ["system", "model"]:
        for k, v in vars(args).items():
            if v is not None and k in config[cfg]:
                config[cfg][k] = v
                args_dic[f"{cfg}:{k}"] = v
    print_dict_as_table(args_dic, "Received parameters from command line (or default):")


def parse_gzip_file(path):
    """Parse gzip file.

    Args:
        path: the file path of gzip file.
    """
    file = gzip.open(path, "rb")
    for line in file:
        yield eval(line)


def get_data_frame_from_gzip_file(path):
    """Get Dataframe from a gzip file.

    Args:
        path the file path of gzip file.

    Returns:
        A dataframe extracted from the gzip file.
    """
    i = 0
    df = {}
    for d in parse_gzip_file(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def save_dataframe_as_npz(data, data_file):
    """Save DataFrame in compressed format.

    Save and convert the DataFrame to npz file.
    Args:
        data (DataFrame): DataFrame to be saved.
        data_file: Target file path.
    """
    user_ids = data[DEFAULT_USER_COL].to_numpy(dtype=data[DEFAULT_USER_COL].dtypes)
    item_ids = data[DEFAULT_ITEM_COL].to_numpy(dtype=data[DEFAULT_ITEM_COL].dtypes)
    ratings = data[DEFAULT_RATING_COL].to_numpy(dtype=np.float32)
    data_dic = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "ratings": ratings,
    }
    if DEFAULT_ORDER_COL in data.columns:
        order_ids = data[DEFAULT_ORDER_COL].to_numpy(dtype=np.long)
        data_dic["order_ids"] = order_ids
    if DEFAULT_TIMESTAMP_COL in data.columns:
        timestamps = data[DEFAULT_TIMESTAMP_COL].to_numpy(dtype=np.long)
        data_dic["timestamps"] = timestamps
    else:
        data_dic["timestamps"] = np.zeros_like(ratings)
    np.savez_compressed(data_file, **data_dic)


def get_dataframe_from_npz(data_file):
    """Get the DataFrame from npz file.

    Get the DataFrame from npz file.

    Args:
        data_file (str or Path): File path.

    Returns:
        DataFrame: the unzip data.
    """
    np_data = np.load(data_file)
    data_dic = {
        DEFAULT_USER_COL: np_data["user_ids"],
        DEFAULT_ITEM_COL: np_data["item_ids"],
        DEFAULT_RATING_COL: np_data["ratings"],
    }
    if "timestamps" in np_data:
        data_dic[DEFAULT_TIMESTAMP_COL] = np_data["timestamps"]
    if "order_ids" in np_data:
        data_dic[DEFAULT_ORDER_COL] = np_data["order_ids"]
    data = pd.DataFrame(data_dic)
    return data


def un_zip(file_name, target_dir=None):
    """Unzip zip files.

    Args:
        file_name (str or Path): zip file path.
        target_dir (str or Path): target path to be save the unzipped files.
    """
    if target_dir is None:
        target_dir = os.path.dirname(file_name)
    zip_file = zipfile.ZipFile(file_name)
    for names in zip_file.namelist():
        print(f"unzip file {names} ...")
        zip_file.extract(names, target_dir)
    zip_file.close()


def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table.

    Args:
        dic (dict): dict object to be formatted.
        tag (str): A name for this dictionary.
        columns ([str,str]):  default ["keys", "values"]. columns name for keys and values.

    Returns:
        None
    """
    print("-" * 80)
    if tag is not None:
        print(tag)
    df = pd.DataFrame(dic.items(), columns=columns)
    print(tabulate(df, headers=columns, tablefmt="psql"))
    print("-" * 80)
    return tabulate(df, headers=columns, tablefmt="psql")


class DictToObject(object):
    """Python dict to object."""

    def __init__(self, dictionary):
        """Initialize DictToObject Class."""

        def _traverse(key, element):
            if isinstance(element, dict):
                return key, DictToObject(element)
            else:
                return key, element

        objd = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(objd)


def get_random_rep(raw_num, dim):
    """Generate a random embedding from a normal (Gaussian) distribution.

    Args:
        raw_num: Number of raw to be generated.
        dim: The dimension of the embeddings.
    Returns:
        ndarray or scalar.
        Drawn samples from the normal distribution.
    """
    return np.random.normal(size=(raw_num, dim))


def timeit(method):
    """Generate decorator for tracking the execution time for the specific method.

    Args:
        method: The method need to timeit.

    To use:
        @timeit
        def method(self):
            pass
    Returns:
        None
    """

    @wraps(method)
    def wrapper(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print(
                "Execute [{}] method costing {:2.2f} ms".format(
                    method.__name__, (te - ts) * 1000
                )
            )
        return result

    return wrapper


def save_to_csv(result, result_file):
    """Save a result dict to disk.

    Args:
        result: The result dict to be saved.
        result_file: The file path to be saved.
    """
    result_df = pd.DataFrame(result)
    if os.path.exists(result_file):
        print(result_file, " already exists, appending result to it")
        total_result = pd.read_csv(result_file)
        total_result = total_result.append(result_df)
    else:
        print("Create new result_file:", result_file)
        total_result = result_df
    total_result.to_csv(result_file, index=False)


def set_seed(seed):
    """Initialize all the seed in the system.

    Args:
        seed: A global random seed.
    """
    if type(seed) != int:
        raise ValueError("Error: seed is invalid type")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    """Convert a string to a bool variable."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
