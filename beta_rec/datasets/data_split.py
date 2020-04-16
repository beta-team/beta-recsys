import numpy as np
import pandas as pd
import math
import os
import sklearn
from tqdm import tqdm
from beta_rec.utils.aliasTable import AliasTable
from beta_rec.utils.constants import *


def filter_by_count(df, group_col, filter_col, num):
    """Filter out the group_col column values that have a less than num count of filter_col

    Args:
        df: DataFrame of interactions
        group_col: column to be filtered
        filter_col: column with the filter condition.
        num: minimum count condition that should be filter out.

    Returns:

    """
    ordercount = (
        df.groupby([group_col])[filter_col].nunique().rename("count").reset_index()
    )
    filter_df = df[
        df[group_col].isin(ordercount[ordercount["count"] >= num][group_col])
    ]
    return filter_df


def filter_user_item(df, min_u_c=5, min_i_c=5):
    """filter data by the minimum purcharce number of items and users

    Args:
        df: DataFrame of interactions
        min_u_c: filter the items that were purchased by less than min_u_c users
        min_i_c: filter the users that have purchased by less than min_i_c items

    Returns:
        The filtered DataFrame

    """
    print(f"filter_user_item under condition min_u_c={min_u_c}, min_i_c={min_i_c}")
    print("-" * 80)
    print("Dataset statistics before filter")
    print(df.agg(["count", "size", "nunique"]))
    n_interact = len(df.index)

    while True:
        # Filter out users that have less than min_i_c interactions (items)
        if min_i_c > 0:
            df = filter_by_count(df, DEFAULT_USER_COL, DEFAULT_ITEM_COL, min_i_c)

        # Filter out items that have less than min_u_c users
        if min_u_c > 0:
            df = filter_by_count(df, DEFAULT_ITEM_COL, DEFAULT_USER_COL, min_u_c)

        new_n_interact = len(df.index)
        if n_interact != new_n_interact:
            n_interact = new_n_interact
        else:
            break  # no change
    print("Dataset statistics after filter")
    print(df.agg(["count", "size", "nunique"]))
    print("-" * 80)
    return df


def filter_user_item_order(df, min_u_c=5, min_i_c=5, min_o_c=5):
    """ filter data by the minimum purcharce number of items and users

    Args:
        df: DataFrame of interactions
        min_u_c: filter the items that were purchased by less than min_u_c users
        min_i_c: filter the users that have purchased by less than min_i_c items
        min_o_c: filter the users that have purchased by less than min_o_c orders

    Returns:
        The filtered DataFrame

    """
    print(
        f"filter_user_item_order under condition min_u_c={min_u_c}, min_i_c={min_i_c}, min_o_c={min_o_c}"
    )
    print("-" * 80)
    print("Dataset statistics before filter")
    print(df.agg(["count", "size", "nunique"]))
    n_interact = len(df.index)

    while True:
        # Filter out users by that have less than min_o_c purchased orders
        if min_o_c > 0:
            df = filter_by_count(df, DEFAULT_USER_COL, DEFAULT_ORDER_COL, min_o_c)

        # Filter out users that have less than min_i_c interactions (items)
        if min_i_c > 0:
            df = filter_by_count(df, DEFAULT_USER_COL, DEFAULT_ITEM_COL, min_i_c)

        # Filter out items that have less than min_u_c users
        if min_u_c > 0:
            df = filter_by_count(df, DEFAULT_ITEM_COL, DEFAULT_USER_COL, min_u_c)

        new_n_interact = len(df.index)
        if n_interact != new_n_interact:
            n_interact = new_n_interact
        else:
            break  # no change
    print("Dataset statistics after filter")
    print(df.agg(["count", "size", "nunique"]))
    print("-" * 80)
    return df


def feed_neg_sample(data, negative_num, item_sampler):
    """feed_neg_sample

    Args:
        data: Dataframe. of interactions.
        negative_num: int. number of negative items.
        item_sampler: polynomial sampler.

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    interact_status = (
        data.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].apply(set).reset_index()
    )
    total_interact = pd.DataFrame(
        {DEFAULT_USER_COL: [], DEFAULT_ITEM_COL: [], DEFAULT_RATING_COL: []},
        dtype=np.long,
    )
    for index, user_items in interact_status.iterrows():
        u = int(user_items[DEFAULT_USER_COL])
        items = set(user_items[DEFAULT_ITEM_COL])  # item set for user u
        n_items = len(items)  # number of positive item for user u
        sample_neg_items = set(
            item_sampler.sample(negative_num + n_items, 1, True)
        )  # first sample negative_num+n_items items
        sample_neg_items = list(sample_neg_items - items)[:negative_num]
        # filter the positive items and truncate the first negative_num
        df_items = np.append(list(items), sample_neg_items)

        df_users = np.array([1] * (negative_num + n_items), dtype=np.long) * u
        df_ones = np.ones(n_items, dtype=np.long)
        df_zeros = np.zeros(negative_num, dtype=np.long)
        ratings = np.append(df_ones, df_zeros)

        df = pd.DataFrame(
            {
                DEFAULT_USER_COL: df_users,
                DEFAULT_ITEM_COL: df_items,
                DEFAULT_RATING_COL: ratings,
            }
        )
        total_interact = total_interact.append(df)

    # shuffle interactions to avoid all the negative samples being together
    total_interact = sklearn.utils.shuffle(total_interact)
    return total_interact


def get_dataframe_from_npz(data_file):
    """Get the DataFrame from npz file

    Get the DataFrame from npz file
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


def load_split_data(path, n_test=10):
    """Load split DataFrame from a specified path

    Args:
        path:
        n_test: number of testing and validation datasets

    Returns:
        train_data: DataFrame of training interaction,
        valid_data_li: DataFrame list of validation interaction,
        test_data_li: DataFrame list of testing interaction,

    """
    train_file = os.path.join(path, "train.npz")
    train_data = get_dataframe_from_npz(train_file)
    print("-" * 80)
    print("train_data statistics")
    print(train_data.agg(["count", "size", "nunique"]))
    valid_data_li = []
    test_data_li = []
    for i in range(n_test):
        print(f"valid_data_{i} statistics")
        valid_df = get_dataframe_from_npz(os.path.join(path, f"valid_{i}.npz"))
        valid_data_li.append(valid_df)
        print(valid_df.agg(["count", "size", "nunique"]))
        test_df = get_dataframe_from_npz(os.path.join(path, f"test_{i}.npz"))
        test_data_li.append(test_df)
        print(f"test_data_{i} statistics")
        print(test_df.agg(["count", "size", "nunique"]))
    print("-" * 80)
    return train_data, valid_data_li, test_data_li


def save_split_data(
    data,
    base_dir,
    data_split="leave_one_basket",
    parameterized_dir=None,
    suffix="train.npz",
):
    """save Dataframe to compressed npz

    Args:
        parameterized_dir: data_split parameter string
        suffix: suffix of the data to be saved
        data (DataFrame): Interactions.
        base_dir: directory to save
        data_split: str. sub folder name to save the data

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

    data_file = os.path.join(base_dir, data_split)
    if not os.path.exists(data_file):
        os.makedirs(data_file)

    data_file = os.path.join(data_file, parameterized_dir)
    if not os.path.exists(data_file):
        os.makedirs(data_file)

    data_file = os.path.join(data_file, suffix)

    np.savez_compressed(data_file, **data_dic)
    print("Data is dumped in :", data_file)


def random_split(data, test_rate=0.1, by_user=False):
    """random_basket_split

    Args:
        data: Dataframe. of interactions.
        test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
        by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    print("random_split")
    data[DEFAULT_FLAG_COL] = "train"
    if by_user:
        users = data[DEFAULT_USER_COL].unique()
        for u in tqdm(users):
            interactions = data[data[DEFAULT_USER_COL] == u].index.values  # numpy array
            interactions = sklearn.utils.shuffle(interactions)
            total_size = len(interactions)
            validate_size = math.ceil(total_size * test_rate)
            test_size = math.ceil(total_size * test_rate)
            train_size = total_size - test_size
            data.loc[
                interactions[train_size:], DEFAULT_FLAG_COL,
            ] = "test"  # the last test_rate of the total orders to be the test set
            data.loc[
                interactions[train_size - validate_size : train_size], DEFAULT_FLAG_COL,
            ] = "validate"

    else:
        interactions = data.index.values  # numpy array
        interactions = sklearn.utils.shuffle(interactions)
        total_size = len(interactions)
        validate_size = math.ceil(total_size * test_rate)
        test_size = math.ceil(total_size * test_rate)
        train_size = total_size - test_size

        data.loc[
            interactions[train_size:], DEFAULT_FLAG_COL,
        ] = "test"  # the last test_rate of the total orders to be the test set
        data.loc[
            interactions[train_size - validate_size : train_size], DEFAULT_FLAG_COL,
        ] = "validate"
    return data


def random_basket_split(data, test_rate=0.1, by_user=False):
    """random_basket_split

    Args:
        data: Dataframe. of interactions.
        test_rate: percentage of the test data. Note that percentage of the vidation data will be the same as testing.
        by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    print("random_basket_split")
    data[DEFAULT_FLAG_COL] = "train"
    if by_user:
        users = data[DEFAULT_USER_COL].unique()
        for u in tqdm(users):
            orders = data[data[DEFAULT_USER_COL] == u][DEFAULT_ORDER_COL].unique()
            orders = sklearn.utils.shuffle(orders)
            total_size = len(orders)
            validate_size = math.ceil(total_size * test_rate)
            test_size = math.ceil(total_size * test_rate)
            train_size = total_size - test_size
            data.loc[
                data[DEFAULT_ORDER_COL].isin(orders[train_size:]), DEFAULT_FLAG_COL,
            ] = "test"  # the last test_rate of the total orders to be the test set
            data.loc[
                data[DEFAULT_ORDER_COL].isin(
                    orders[train_size - validate_size : train_size]
                ),
                DEFAULT_FLAG_COL,
            ] = "validate"

    else:
        orders = data[DEFAULT_ORDER_COL].unique()
        orders = sklearn.utils.shuffle(orders)
        total_size = len(orders)
        validate_size = math.ceil(total_size * test_rate)
        test_size = math.ceil(total_size * test_rate)
        train_size = total_size - test_size
        data.loc[
            data[DEFAULT_ORDER_COL].isin(orders[train_size:]), DEFAULT_FLAG_COL,
        ] = "test"  # the last test_rate of the total orders to be the test set
        data.loc[
            data[DEFAULT_ORDER_COL].isin(
                orders[train_size - validate_size : train_size]
            ),
            DEFAULT_FLAG_COL,
        ] = "validate"
    return data


def leave_one_out(data, random=False):
    """leave_one_out split

    Args:
        data: Dataframe. of interactions.
        random: bool.  Whether randomly leave one item/basket as testing. only for leave_one_out and leave_one_basket

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    print("leave_one_out")
    data = filter_user_item(data, min_u_c=0, min_i_c=3)
    data[DEFAULT_FLAG_COL] = "train"
    if random:
        data = sklearn.utils.shuffle(data)

    users = data[DEFAULT_USER_COL].unique()
    for u in tqdm(users):
        interactions = data[data[DEFAULT_USER_COL] == u].index.values
        data.loc[interactions[-1], DEFAULT_FLAG_COL] = "test"
        data.loc[interactions[-2], DEFAULT_FLAG_COL] = "validate"

    return data


def leave_one_basket(data, random=False):
    """leave_one_basket split

    Args:
        data: Dataframe. of interactions.
        random: bool.  Whether randomly leave one item/basket as testing. only for leave_one_out and leave_one_basket

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    print("leave_one_basket")
    data[DEFAULT_FLAG_COL] = "train"
    if random:
        data = sklearn.utils.shuffle(data)
    else:
        data.sort_values(by=[DEFAULT_TIMESTAMP_COL], inplace=True)

    users = data[DEFAULT_USER_COL].unique()
    for u in tqdm(users):
        user_orders = data[data[DEFAULT_USER_COL] == u][DEFAULT_ORDER_COL].unique()
        data.loc[data[DEFAULT_ORDER_COL] == user_orders[-1], DEFAULT_FLAG_COL] = "test"
        data.loc[
            data[DEFAULT_ORDER_COL] == user_orders[-2], DEFAULT_FLAG_COL
        ] = "validate"
    return data


def temporal_split(data, test_rate=0.1, by_user=False):
    """temporal_split

    Args:
        data: Dataframe. of interactions.
        test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
        by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    print("temporal_split")
    data[DEFAULT_FLAG_COL] = "train"
    data.sort_values(by=[DEFAULT_TIMESTAMP_COL], inplace=True)
    if by_user:
        users = data[DEFAULT_USER_COL].unique()
        for u in tqdm(users):
            interactions = data[data[DEFAULT_USER_COL] == u].index.values
            total_size = len(interactions)
            validate_size = math.ceil(total_size * test_rate)
            test_size = math.ceil(total_size * test_rate)
            train_size = total_size - test_size

            data.loc[
                interactions[train_size:], DEFAULT_FLAG_COL,
            ] = "test"  # the last test_rateof the total orders to be the test set
            data.loc[
                interactions[train_size - validate_size : train_size], DEFAULT_FLAG_COL,
            ] = "validate"

    else:
        interactions = data.index.values
        total_size = len(interactions)
        validate_size = math.ceil(total_size * test_rate)
        test_size = math.ceil(total_size * test_rate)
        train_size = total_size - test_size

        data.loc[
            interactions[train_size:], DEFAULT_FLAG_COL,
        ] = "test"  # the last test_rate of the total orders to be the test set
        data.loc[
            interactions[train_size - validate_size : train_size], DEFAULT_FLAG_COL,
        ] = "validate"
    return data


def temporal_basket_split(data, test_rate=0.1, by_user=False):
    """temporal_basket_split

    Args:
        data: Dataframe. of interactions. must have a col DEFAULT_ORDER_COL
        test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
        by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    print("temporal_split_basket")
    data[DEFAULT_FLAG_COL] = "train"
    data.sort_values(by=[DEFAULT_TIMESTAMP_COL], inplace=True)
    if by_user:
        users = data[DEFAULT_USER_COL].unique()
        for u in tqdm(users):
            orders = data[data[DEFAULT_USER_COL] == u][DEFAULT_ORDER_COL].unique()
            total_size = len(orders)
            validate_size = math.ceil(total_size * test_rate)
            test_size = math.ceil(total_size * test_rate)
            train_size = total_size - test_size
            data.loc[
                data[DEFAULT_ORDER_COL].isin(orders[train_size:]), DEFAULT_FLAG_COL,
            ] = "test"  # the last test_rate of the total orders to be the test set
            data.loc[
                data[DEFAULT_ORDER_COL].isin(
                    orders[train_size - validate_size : train_size]
                ),
                DEFAULT_FLAG_COL,
            ] = "validate"
    else:
        orders = data[DEFAULT_ORDER_COL].unique()
        total_size = len(orders)
        validate_size = math.ceil(total_size * test_rate)
        test_size = math.ceil(total_size * test_rate)
        train_size = total_size - test_size
        data.loc[
            data[DEFAULT_ORDER_COL].isin(orders[train_size:]), DEFAULT_FLAG_COL,
        ] = "test"  # the last test_rate of the total orders to be the test set
        data.loc[
            data[DEFAULT_ORDER_COL].isin(
                orders[train_size - validate_size : train_size]
            ),
            DEFAULT_FLAG_COL,
        ] = "validate"
    return data


def split_data(
    data,
    split_type,
    test_rate,
    random=False,
    n_negative=100,
    save_dir=None,
    by_user=False,
    n_test=10,
):
    """Data split methods

    Args:
        data: Dataframe. of interactions.
        split_type: str. options can be:
                        - random
                        - random_basket
                        - leave_one_out
                        - leave_one_basket
                        - temporal
                        - temporal_basket
        random: bool. Whether random leave one item/basket as testing. only for leave_one_out and leave_one_basket
        test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
        n_negative: Number of negative samples for testing and validation data.
        save_dir: str. Default None. If specified, the split data will be saved to the dir.
        by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
        n_test: int. Default 10. The number of testing and validation copies.

    Returns:
        Dataframe. The split data. Note that the returned data will not have negative samples.

    """
    print(f"Splitting data by {split_type} ...")
    if split_type == "random":
        data = random_split(data, test_rate, by_user)
    elif split_type == "random_basket":
        data = random_basket_split(data, test_rate, by_user)
    elif split_type == "leave_one_out":
        data = leave_one_out(data, random)
    elif split_type == "leave_one_basket":
        data = leave_one_basket(data, random)
    elif split_type == "temporal":
        data = temporal_split(data, test_rate, by_user)
    elif split_type == "temporal_basket":
        data = temporal_basket_split(data, test_rate, by_user)
    else:
        print("[ERROR] wrong split_type.")
        return None
    tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
    tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
    tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
    if save_dir is None:
        return data

    parameterized_path = generate_parameterized_path(
        test_rate=test_rate, random=random, n_negative=n_negative, by_user=by_user
    )

    save_split_data(tp_train, save_dir, split_type, parameterized_path, "train.npz")
    item_sampler = AliasTable(data[DEFAULT_ITEM_COL].value_counts().to_dict())
    for i in range(n_test):
        tp_validate_new = feed_neg_sample(tp_validate, n_negative, item_sampler)
        tp_test_new = feed_neg_sample(tp_test, n_negative, item_sampler)
        save_split_data(
            tp_validate_new,
            save_dir,
            split_type,
            parameterized_path,
            "valid_" + str(i) + ".npz",
        )
        save_split_data(
            tp_test_new,
            save_dir,
            split_type,
            parameterized_path,
            "test_" + str(i) + ".npz",
        )
    return data


def generate_random_data(n_interaction, user_id, item_id):
    oder_id = 10
    users = np.random.randint(user_id, size=n_interaction)
    orders = np.random.randint(oder_id, size=n_interaction) * 100 + users
    timestamps = orders
    items = np.random.randint(item_id, size=n_interaction)
    ratings = np.array([1] * n_interaction)

    data = {
        DEFAULT_USER_COL: users,
        DEFAULT_ORDER_COL: orders,
        DEFAULT_TIMESTAMP_COL: timestamps,
        DEFAULT_ITEM_COL: items,
        DEFAULT_RATING_COL: ratings,
    }
    data = pd.DataFrame(data)
    return data


def generate_parameterized_path(
    test_rate=0, random=False, n_negative=100, by_user=False
):
    """Generate parameterized path.

    Encode parameters into path to differentiate different split parameters

    Args:
        by_user: split by user
        test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
        random: bool. Whether random leave one item/basket as testing. only for leave_one_out and leave_one_basket
        n_negative: Number of negative samples for testing and validation data.

    Returns:
        A string that encodes parameters.
    """
    path_str = ""
    if by_user:
        path_str = "by_user" + path_str
    else:
        path_str = "global" + path_str
    test_rate *= 100
    test_rate = round(test_rate)
    path_str += "_test_rate_" + str(test_rate) if test_rate != 0 else ""
    path_str += "_random" if random is True else ""
    path_str += "_n_neg_" + str(n_negative)
    return path_str
