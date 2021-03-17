import math
import os
import time

import numpy as np
import pandas as pd
import sklearn
from tabulate import tabulate
from tqdm import tqdm

from ..utils.alias_table import AliasTable
from ..utils.common_util import get_dataframe_from_npz, save_dataframe_as_npz
from ..utils.constants import (
    DEFAULT_FLAG_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_ORDER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_USER_COL,
)


def filter_by_count(df, group_col, filter_col, num):
    """Filter out the group_col column values that have a less than num count of filter_col.

    Args:
        df (DataFrame): interaction DataFrame to be processed.
        group_col (string): column name to be filtered.
        filter_col (string): column with the filter condition.
        num (int): minimum count condition that should be filter out.

    Returns:
        DataFrame: The filtered interactions.

    """
    ordercount = (
        df.groupby([group_col])[filter_col].nunique().rename("count").reset_index()
    )
    filter_df = df[
        df[group_col].isin(ordercount[ordercount["count"] >= num][group_col])
    ]
    return filter_df


def check_data_available(data):
    """Check if a dataset is available after filtering.

    Check whether a given dataset is available for later use.

    Args:
        data (DataFrame): interaction DataFrame to be processed.

    Raises:
        RuntimeError: An error occurred it there is no interaction.

    """
    if len(data.index) < 1:
        raise RuntimeError(
            "This dataset contains no interaction after filtering. Please check the default filter setup of this split!"
        )


def filter_user_item(df, min_u_c=5, min_i_c=5):
    """Filter data by the minimum purchase number of items and users.

    Args:
        df (DataFrame): interaction DataFrame to be processed.
        min_u_c (int): filter the items that were purchased by less than min_u_c users.
            (default: :obj:`5`)
        min_i_c (int): filter the users that have purchased by less than min_i_c items.
            (default: :obj:`5`)

    Returns:
        DataFrame: The filtered interactions

    """
    print(f"filter_user_item under condition min_u_c={min_u_c}, min_i_c={min_i_c}")
    print("-" * 80)
    print("Dataset statistics before filter")
    print(
        tabulate(
            df.agg(["count", "nunique"]),
            headers=df.columns,
            tablefmt="psql",
            disable_numparse=True,
        )
    )
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
    check_data_available(df)
    print("Dataset statistics after filter")
    print(
        tabulate(
            df.agg(["count", "nunique"]),
            headers=df.columns,
            tablefmt="psql",
            disable_numparse=True,
        )
    )
    print("-" * 80)
    return df


def filter_user_item_order(df, min_u_c=5, min_i_c=5, min_o_c=5):
    """Filter data by the minimum purchase number of items and users.

    Args:
        df (DataFrame): interaction DataFrame to be processed.
        min_u_c: filter the items that were purchased by less than min_u_c users.
        (default: :obj:`5`)
        min_i_c: filter the users that have purchased by less than min_i_c items.
        (default: :obj:`5`)
        min_o_c: filter the users that have purchased by less than min_o_c orders.
        (default: :obj:`5`)

    Returns:
        The filtered DataFrame.
    """
    print(
        f"filter_user_item_order under condition min_u_c={min_u_c}, min_i_c={min_i_c}, min_o_c={min_o_c}"
    )
    print("-" * 80)
    print("Dataset statistics before filter")
    print(
        tabulate(
            df.agg(["count", "nunique"]),
            headers=df.columns,
            tablefmt="psql",
            disable_numparse=True,
        )
    )
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
    check_data_available(df)
    print("Dataset statistics after filter")
    print(
        tabulate(
            df.agg(["count", "nunique"]),
            headers=df.columns,
            tablefmt="psql",
            disable_numparse=True,
        )
    )
    print("-" * 80)
    return df


def feed_neg_sample(data, negative_num, item_sampler):
    """Sample negative items for a interaction DataFrame.

    Args:
        data (DataFrame): interaction DataFrame to be processed.
        negative_num (int): number of negative items.
                        if negative_num<0, will keep all the negative items for each user.
        item_sampler (AliasTable): a AliasTable sampler that contains the items.
    Returns:
        DataFrame: interaction DataFrame with a new 'flag' column labeling with "train", "test" or "valid".
    """
    unique_item_set = set(data[DEFAULT_ITEM_COL].unique())
    unique_rating_num = data[DEFAULT_RATING_COL].nunique()
    interact_status = (
        data.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].apply(set).reset_index()
    )
    total_interact = pd.DataFrame(
        {DEFAULT_USER_COL: [], DEFAULT_ITEM_COL: [], DEFAULT_RATING_COL: []},
        dtype=np.long,
    )
    for index, user_items in interact_status.iterrows():
        u = user_items[DEFAULT_USER_COL]
        pos_items = set(user_items[DEFAULT_ITEM_COL])  # item set for user u
        pos_items_li = list(pos_items)  # the positive items should be unique
        n_pos_items = len(pos_items_li)  # number of positive item for user u

        if negative_num < 0:  # keep all the negative items
            neg_items_li = list(unique_item_set - pos_items)
            n_neg_items = len(neg_items_li)
        else:  # only keep negative_num negative items
            n_neg_items = negative_num
            neg_items = set(item_sampler.sample(negative_num + n_pos_items, 1, True))
            neg_items_li = list(neg_items - pos_items)[:negative_num]
        # filter the positive items and truncate the first negative_num
        df_items = np.append(pos_items_li, neg_items_li)
        df_users = np.array([u] * (n_pos_items + n_neg_items), dtype=type(u))
        pos_rating = []

        if unique_rating_num != 1:
            # get the rating scores.
            for item in pos_items_li:
                pos_rating.append(
                    data.loc[
                        (data[DEFAULT_USER_COL] == u)
                        & (data[DEFAULT_ITEM_COL] == item),
                        DEFAULT_RATING_COL,
                    ].to_numpy()[0]
                )
        else:
            pos_rating = np.full(n_pos_items, 1)
        neg_rating = np.zeros(n_neg_items, dtype=np.long)
        df_ratings = np.append(pos_rating, neg_rating)

        df = pd.DataFrame(
            {
                DEFAULT_USER_COL: df_users,
                DEFAULT_ITEM_COL: df_items,
                DEFAULT_RATING_COL: df_ratings,
            }
        )
        total_interact = total_interact.append(df)
    # shuffle interactions to avoid all the negative samples being together
    total_interact = sklearn.utils.shuffle(total_interact)
    return total_interact


def load_split_data(path, n_test=10):
    """Load split DataFrame from a specified path.

    Args:
        path (string): split data path.
        n_test: number of testing and validation datasets.
                If n_test==0, will load the original (no negative items) valid and test datasets.

    Returns:
        (DataFrame, list(DataFrame), list(DataFrame)): DataFrame of training interaction,
        DataFrame list of validation interaction,
        DataFrame list of testing interaction,
    """
    train_file = os.path.join(path, "train.npz")
    train_data = get_dataframe_from_npz(train_file)
    print("-" * 80)
    print("Loaded training set statistics")
    print(
        tabulate(
            train_data.agg(["count", "nunique"]),
            headers=train_data.columns,
            tablefmt="psql",
            disable_numparse=True,
        )
    )
    if not n_test:
        valid_df = get_dataframe_from_npz(os.path.join(path, "valid.npz"))
        test_df = get_dataframe_from_npz(os.path.join(path, "test.npz"))
        print("Loaded validation set statistics")
        print(
            tabulate(
                valid_df.agg(["count", "nunique"]),
                headers=valid_df.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )
        print("Loaded testing set statistics")
        print(
            tabulate(
                test_df.agg(["count", "nunique"]),
                headers=test_df.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )
        print("-" * 80)
        return train_data, valid_df, test_df

    valid_data_li = []
    test_data_li = []
    for i in range(n_test):
        valid_df = get_dataframe_from_npz(os.path.join(path, f"valid_{i}.npz"))
        valid_data_li.append(valid_df)
        if i == 0:
            print(f"valid_data_{i} statistics")
            print(
                tabulate(
                    valid_df.agg(["count", "nunique"]),
                    headers=valid_df.columns,
                    tablefmt="psql",
                    disable_numparse=True,
                )
            )
        test_df = get_dataframe_from_npz(os.path.join(path, f"test_{i}.npz"))
        test_data_li.append(test_df)
        if i == 0:
            print(f"test_data_{i} statistics")
            print(
                tabulate(
                    test_df.agg(["count", "nunique"]),
                    headers=test_df.columns,
                    tablefmt="psql",
                    disable_numparse=True,
                )
            )
    print("-" * 80)
    return train_data, valid_data_li, test_data_li


def save_split_data(
    data,
    base_dir,
    data_split="leave_one_basket",
    parameterized_dir=None,
    suffix="train.npz",
):
    """Save DataFrame to compressed npz.

    Args:
        data (DataFrame): interaction DataFrame to be saved.
        parameterized_dir (string): data_split parameter string.
        suffix (string): suffix of the data to be saved.
        base_dir (string): directory to save.
        data_split (string): sub folder name for saving the data.
    """
    data_file = os.path.join(base_dir, data_split)
    if not os.path.exists(data_file):
        os.makedirs(data_file)

    data_file = os.path.join(data_file, parameterized_dir)
    if not os.path.exists(data_file):
        os.makedirs(data_file)

    data_file = os.path.join(data_file, suffix)

    save_dataframe_as_npz(data, data_file)
    print("Data is dumped in :", data_file)


def random_split(data, test_rate=0.1, by_user=False):
    """random_basket_split.

    Args:
        data (DataFrame): interaction DataFrame to be split.
        test_rate (float): percentage of the test data.
            Note that percentage of the validation data will be the same as testing.
        by_user (bool): Default False.
                    - Ture: user-based split,
                    - False: global split,

    Returns:
        DataFrame: DataFrame that have already by labeled by a col with "train", "test" or "valid".
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
                interactions[train_size:],
                DEFAULT_FLAG_COL,
            ] = "test"  # the last test_rate of the total orders to be the test set
            data.loc[
                interactions[train_size - validate_size : train_size],
                DEFAULT_FLAG_COL,
            ] = "validate"

    else:
        interactions = data.index.values  # numpy array
        interactions = sklearn.utils.shuffle(interactions)
        total_size = len(interactions)
        validate_size = math.ceil(total_size * test_rate)
        test_size = math.ceil(total_size * test_rate)
        train_size = total_size - test_size

        data.loc[
            interactions[train_size:],
            DEFAULT_FLAG_COL,
        ] = "test"  # the last test_rate of the total orders to be the test set
        data.loc[
            interactions[train_size - validate_size : train_size],
            DEFAULT_FLAG_COL,
        ] = "validate"
    return data


def random_basket_split(data, test_rate=0.1, by_user=False):
    """random_basket_split.

    Args:
        data (DataFrame): interaction DataFrame to be split.
        test_rate (float): percentage of the test data.
            Note that percentage of the validation data will be the same as testing.
        by_user (bool): Default False.
                    - True: user-based split,
                    - False: global split,

    Returns:
        DataFrame: DataFrame that have already by labeled by a col with "train", "test" or "valid".
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
                data[DEFAULT_ORDER_COL].isin(orders[train_size:]),
                DEFAULT_FLAG_COL,
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
            data[DEFAULT_ORDER_COL].isin(orders[train_size:]),
            DEFAULT_FLAG_COL,
        ] = "test"  # the last test_rate of the total orders to be the test set
        data.loc[
            data[DEFAULT_ORDER_COL].isin(
                orders[train_size - validate_size : train_size]
            ),
            DEFAULT_FLAG_COL,
        ] = "validate"
    return data


def leave_one_out(data, random=False):
    """leave_one_out split.

    Args:
        data (DataFrame): interaction DataFrame to be split.
        random (bool):  Whether randomly leave one item/basket as testing. only for leave_one_out and leave_one_basket.

    Returns:
        DataFrame: DataFrame that have already by labeled by a col with "train", "test" or "valid".
    """
    start_time = time.time()
    print("leave_one_out")
    data[DEFAULT_FLAG_COL] = "train"
    if random:
        data = sklearn.utils.shuffle(data)
    else:
        data.sort_values(by=[DEFAULT_TIMESTAMP_COL], ascending=False, inplace=True)

    data.loc[
        data.groupby([DEFAULT_USER_COL]).head(2).index, DEFAULT_FLAG_COL
    ] = "validate"
    data.loc[data.groupby([DEFAULT_USER_COL]).head(1).index, DEFAULT_FLAG_COL] = "test"

    end_time = time.time()
    print(f"leave_one_out time cost: {end_time - start_time}")
    return data


def leave_one_basket(data, random=False):
    """leave_one_basket split.

    Args:
        data (DataFrame): interaction DataFrame to be split.
        random (bool):  Whether randomly leave one item/basket as testing. only for leave_one_out and leave_one_basket.

    Returns:
        DataFrame: DataFrame that have already by labeled by a col with "train", "test" or "valid".
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
    """temporal_split.

    Args:
        data (DataFrame): interaction DataFrame to be split.
        test_rate (float): percentage of the test data.
            Note that percentage of the validation data will be the same as testing.
        by_user (bool): bool. Default False.
                    - True: user-based split,
                    - False: global split,

    Returns:
        DataFrame: DataFrame that have already by labeled by a col with "train", "test" or "valid".
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
                interactions[train_size:],
                DEFAULT_FLAG_COL,
            ] = "test"  # the last test_rate of the total orders to be the test set
            data.loc[
                interactions[train_size - validate_size : train_size],
                DEFAULT_FLAG_COL,
            ] = "validate"

    else:
        interactions = data.index.values
        total_size = len(interactions)
        validate_size = math.ceil(total_size * test_rate)
        test_size = math.ceil(total_size * test_rate)
        train_size = total_size - test_size

        data.loc[
            interactions[train_size:],
            DEFAULT_FLAG_COL,
        ] = "test"  # the last test_rate of the total orders to be the test set
        data.loc[
            interactions[train_size - validate_size : train_size],
            DEFAULT_FLAG_COL,
        ] = "validate"
    return data


def temporal_basket_split(data, test_rate=0.1, by_user=False):
    """temporal_basket_split.

    Args:
        data (DataFrame): interaction DataFrame to be split.
            It must have a col DEFAULT_ORDER_COL.
        test_rate (float): percentage of the test data.
            Note that percentage of the validation data will be the same as testing.
        by_user (bool): Default False.
                    - True: user-based split,
                    - False: global split,

    Returns:
        DataFrame: DataFrame that have already by labeled by a col with "train", "test" or "valid".
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
                data[DEFAULT_ORDER_COL].isin(orders[train_size:]),
                DEFAULT_FLAG_COL,
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
            data[DEFAULT_ORDER_COL].isin(orders[train_size:]),
            DEFAULT_FLAG_COL,
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
    """Split data by split_type and other parameters.

    Args:
        data (DataFrame): interaction DataFrame to be split
        split_type (string): options can be:
                        - random
                        - random_basket
                        - leave_one_out
                        - leave_one_basket
                        - temporal
                        - temporal_basket
        random (bool): Whether random leave one item/basket as testing. only for leave_one_out and leave_one_basket.
        test_rate (float): percentage of the test data.
            Note that percentage of the validation data will be the same as testing.
        n_negative (int): Number of negative samples for testing and validation data.
        save_dir (string or Path): Default None. If specified, the split data will be saved to the dir.
        by_user (bool): Default False.
                    - True: user-based split,
                    - False: global split,
        n_test (int): Default 10. The number of testing and validation copies.

    Returns:
        DataFrame: The split data. Note that the returned data will not have negative samples.

    """
    print(f"Splitting data by {split_type} ...")
    if n_negative < 0 and n_test > 1:
        # n_negative < 0, validate and testing sets of splits will contain all the negative items.
        # There will be only one validate and one testing sets.
        n_test = 1
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
    # keep the original validation and test sets.
    save_split_data(tp_validate, save_dir, split_type, parameterized_path, "valid.npz")
    save_split_data(tp_test, save_dir, split_type, parameterized_path, "test.npz")
    item_sampler = AliasTable(data[DEFAULT_ITEM_COL].value_counts().to_dict())
    n_items = tp_train[DEFAULT_ITEM_COL].nunique()
    valid_neg_max = (
        tp_validate.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].count().max()
    )
    test_neg_max = tp_test.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].count().max()
    if n_items - valid_neg_max < n_negative or n_items - test_neg_max < n_negative:
        raise RuntimeError(
            "This dataset do not have sufficient negative items for sampling! \n"
            + f"valid_neg_max: {n_items - valid_neg_max}, "
            + f"test_neg_max: {n_items - test_neg_max},"
            + f"n_negative: {n_negative}\nPlease directly use valid.npz and test.npz."
        )
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
    """Generate random data for testing.

    Generate random data for unit test.
    """
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

    Encode parameters into path to differentiate different split parameters.

    Args:
        by_user (bool): split by user.
        test_rate (float): percentage of the test data.
            Note that percentage of the validation data will be the same as testing.
        random (bool): Whether random leave one item/basket as testing. only for leave_one_out and leave_one_basket.
        n_negative (int): Number of negative samples for testing and validation data.

    Returns:
        string: A string that encodes parameters.
    """
    path_str = ""
    if by_user:
        path_str = "user_based" + path_str
    else:
        path_str = "full" + path_str
    test_rate *= 100
    test_rate = round(test_rate)
    path_str += "_test_rate_" + str(test_rate) if test_rate != 0 else ""
    path_str += "_random" if random is True else ""
    path_str += "_n_neg_" + str(n_negative)
    return path_str
