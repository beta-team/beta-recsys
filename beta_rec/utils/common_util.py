import os
import random
import time
import numpy as np
import torch
import pandas as pd
import zipfile
from beta_rec.utils.constants import *


def update_args(config, args):
    """Update config parameters by the received parameters from command line

    Args:
        config (dict): Initial dict of the parameters from JOSN config file.
        args (object): An argparse Argument object with attributes being the parameters to be updated.

    Returns:
        None
    """
    print("Received parameters form command line:")
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
            print(k, "\t", v)


def save_dataframe_as_npz(data, data_file):
    """ Save DataFrame in compressed format
    Save and convert the DataFrame to npz file.
    Args:
        data (DataFrame): DataFrame to be saved
        data_file: Target file path

    Returns:

    """
    user_ids = data[DEFAULT_USER_COL].to_numpy(dtype=np.long)
    item_ids = data[DEFAULT_ITEM_COL].to_numpy(dtype=np.long)
    ratings = data[DEFAULT_RATING_COL].to_numpy(dtype=np.float32)
    columns_dic = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "ratings": ratings,
    }
    if DEFAULT_ORDER_COL in data.columns:
        order_ids = data[DEFAULT_ORDER_COL].to_numpy(dtype=np.long)
        columns_dic["order_ids"] = order_ids
    if DEFAULT_TIMESTAMP_COL in data.columns:
        timestamps = data[DEFAULT_TIMESTAMP_COL].to_numpy(dtype=np.long)
        columns_dic["timestamps"] = timestamps
    np.savez_compressed(data_file, **columns_dic)


def get_dataframe_from_npz(data_file):
    """Get the DataFrame from npz file

    Get the DataFrame from npz file

    Args:
        data_file: File path

    Returns:
        DataFrame
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
        file_name: zip file path.

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


def dict2str(tag, dic):
    """
    A better format to print a dictionary
    Args:
        tag: str. A name for this dictionary
        dic: dict.

    Returns:
        None
    """

    dic_str = (
        tag
        + ": \n"
        + "\n".join([str(k) + ":\t" + str(v) for k, v in dic.items()])
        + "\n"
    )
    print("-" * 80)
    print(dic_str)
    print("-" * 80)
    return dic_str


def initialize_folders():
    """
    Initialize the whole directory structure of the project
    Returns:
        None

    """
    utils_root = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(utils_root, "..", ".."))

    configs = base_dir + "/configs/"
    datasets = base_dir + "/datasets/"
    checkpoints = base_dir + "/checkpoints/"
    results = base_dir + "/results/"
    logs = base_dir + "/logs/"
    samples = base_dir + "/samples/"
    runs = base_dir + "/runs/"

    for dir in [configs, datasets, checkpoints, results, samples, logs, runs]:
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
    """
    Decorator for tracking the execution time for the specific method
    Args:
        method: The method need to timeit.

    To use:
        @timeit
        def method(self):
            pass
    Returns:
        None
    
    """

    def timed(*args, **kw):
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

    return timed


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
