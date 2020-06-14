import os
import random
import time
import zipfile
from functools import wraps

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from beta_rec.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_ORDER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)


def ensureDir(dir_path):
    """Ensure a dir exist, otherwise create

    Args:
        dir_path (str): the target dir
    Return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def update_args(config, args):
    """Update config parameters by the received parameters from command line

    Args:
        config (dict): Initial dict of the parameters from JSON config file.
        args (object): An argparse Argument object with attributes being the parameters to be updated.

    Returns:
        None
    """
    args_dic = {}
    for cfg in ["system", "model"]:
        for k, v in vars(args).items():
            if v is not None and k in config[cfg]:
                config[cfg][k] = v
                args_dic[f"{cfg}:{k}"] = v
    print_dict_as_table(args_dic, "Received parameters form command line (or default):")


def save_dataframe_as_npz(data, data_file):
    """ Save DataFrame in compressed format
    Save and convert the DataFrame to npz file.
    Args:
        data (DataFrame): DataFrame to be saved
        data_file: Target file path

    Returns:
        None
    """
    user_ids = data[DEFAULT_USER_COL].to_numpy(dtype=np.long)
    item_ids = data[DEFAULT_ITEM_COL].to_numpy(dtype=np.long)
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
    """Get the DataFrame from npz file

    Get the DataFrame from npz file

    Args:
        data_file (str or Path): File path

    Returns:
        DataFrame:

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
    """ Unzip zip files

    Args:
        file_name (str or Path): zip file path.
        target_dir (str or Path): target path to be save the unzipped files.

    Returns:
        None

    """
    if target_dir is None:
        target_dir = os.path.dirname(file_name)
    zip_file = zipfile.ZipFile(file_name)
    for names in zip_file.namelist():
        print(f"unzip file {names} ...")
        zip_file.extract(names, target_dir)
    zip_file.close()


def print_dict_as_table(dic, tag=None, columns=["keys", "values"]):
    """Print a dictionary as table

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
    """Python dict to object

    """

    def __init__(self, dictionary):
        def _traverse(key, element):
            if isinstance(element, dict):
                return key, DictToObject(element)
            else:
                return key, element

        objd = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(objd)


def initialize_folders(base_dir):
    """ Initialize the whole directory structure of the project

    Args:
        base_dir (str): Root path of the project.

    Returns:
        None
    """

    configs = base_dir + "/configs/"
    datasets = base_dir + "/datasets/"
    checkpoints = base_dir + "/checkpoints/"
    results = base_dir + "/results/"
    logs = base_dir + "/logs/"
    processes = base_dir + "/processes/"
    runs = base_dir + "/runs/"

    for dir in [configs, datasets, checkpoints, results, processes, logs, runs]:
        if not os.path.exists(dir):
            os.makedirs(dir)


def get_random_rep(raw_num, dim):
    """
    Generate a random embedding from a normal (Gaussian) distribution.
    Args:
        raw_num: Number of raw to be generated.
        dim: The dimension of the embeddings.
    Returns:
        ndarray or scalar
        Drawn samples from the normal distribution.
    """
    return np.random.normal(size=(raw_num, dim))


def timeit(method):
    """Decorator for tracking the execution time for the specific method
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
    """
    Save a result dict to disk.

    Args:
        result: The result dict to be saved.
        result_file: The file path to be saved.

    Returns:
        None

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
    """
    Initialize all the seed in the system

    Args:
        seed: A global random seed.

    Returns:
        None

    """
    if type(seed) != int:
        raise ValueError("Error: seed is invalid type")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
