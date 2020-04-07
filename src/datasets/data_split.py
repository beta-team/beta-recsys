import numpy as np
import pandas as pd
import math
import os
import sklearn
from tqdm import tqdm
from src.utils.unigramTable import UnigramTable
from src.utils.constants import *


def fiter_by_count(df, group_col, filter_col, num):
    ordercount = (
        df.groupby([group_col])[filter_col].nunique().rename("count").reset_index()
    )
    filter_df = df[
        df[group_col].isin(ordercount[ordercount["count"] >= num][group_col])
    ]
    return filter_df


def filter_user_item(df, min_u_c=5, min_i_c=5):
    n_interact = len(df.index)
    n_users = df[DEFAULT_USER_COL].nunique()
    n_items = df[DEFAULT_ITEM_COL].nunique()
    print(
        f"before filter, n_interact:{n_interact}, n_users:{n_users}, n_items:{n_items}"
    )

    while True:
        # Filter out users that have less than min_i_c interactions (items)
        if min_i_c > 0:
            df = fiter_by_count(df, DEFAULT_USER_COL, DEFAULT_ITEM_COL, min_i_c)

        # Filter out items that have less than min_u_c users
        if min_u_c > 0:
            df = fiter_by_count(df, DEFAULT_ITEM_COL, DEFAULT_USER_COL, min_u_c)

        new_n_interact = len(df.index)
        if n_interact != new_n_interact:
            n_interact = new_n_interact
        else:
            break  # no change

    n_interact = len(df.index)
    n_orders = df[DEFAULT_ORDER_COL].nunique()
    n_users = df[DEFAULT_USER_COL].nunique()
    n_items = df[DEFAULT_ITEM_COL].nunique()
    print("after filter", n_interact, n_orders, n_users, n_items)
    return df


# filter data by the minimum purcharce number of items and users
def filter_user_item_order(df, min_u_c=5, min_i_c=5, min_o_c=5):
    n_interact = len(df.index)
    n_orders = df[DEFAULT_ORDER_COL].nunique()
    n_users = df[DEFAULT_USER_COL].nunique()
    n_items = df[DEFAULT_ITEM_COL].nunique()
    print("before filter", n_interact, n_orders, n_users, n_items)

    while True:
        # Filter out users by that have less than min_o_c purchased orders
        if min_o_c > 0:
            df = fiter_by_count(df, DEFAULT_USER_COL, DEFAULT_ORDER_COL, min_o_c)

        # Filter out users that have less than min_i_c interactions (items)
        if min_i_c > 0:
            df = fiter_by_count(df, DEFAULT_USER_COL, DEFAULT_ITEM_COL, min_i_c)

        # Filter out items that have less than min_u_c users
        if min_u_c > 0:
            df = fiter_by_count(df, DEFAULT_ITEM_COL, DEFAULT_USER_COL, min_u_c)

        new_n_interact = len(df.index)
        if n_interact != new_n_interact:
            n_interact = new_n_interact
        else:
            break  # no change

    n_interact = len(df.index)
    n_orders = df[DEFAULT_ORDER_COL].nunique()
    n_users = df[DEFAULT_USER_COL].nunique()
    n_items = df[DEFAULT_ITEM_COL].nunique()
    print("after filter", n_interact, n_orders, n_users, n_items)
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


def save_data(data, base_dir, data_split="leave_one_basket", suffix="train.npz"):
    """save Dataframe to compressed npz

    Args:
        data: Dataframe. of interactions.
        base_dir: directory to save
        data_split: str. sub folder name to save the data

    Returns: Dataframe that have already by labeled by a col with "train", "test" or "valid".
    """
    user_ids = data[DEFAULT_USER_COL].to_numpy(dtype=np.long)
    item_ids = data[DEFAULT_ITEM_COL].to_numpy(dtype=np.long)
    if DEFAULT_ORDER_COL in data.columns:
        order_ids = data[DEFAULT_ORDER_COL].to_numpy(dtype=np.long)
    ratings = data[DEFAULT_RATING_COL].to_numpy(dtype=np.float32)
    if DEFAULT_TIMESTAMP_COL in data.columns:
        timestamps = data[DEFAULT_TIMESTAMP_COL].to_numpy(dtype=np.long)
    else:
        timestamps = np.zeros_like(ratings)

    data_file = os.path.join(base_dir, data_split)
    if not os.path.exists(data_file):
        os.makedirs(data_file)
    data_file = os.path.join(data_file, suffix)

    if DEFAULT_ORDER_COL in data.columns:
        np.savez_compressed(
            data_file,
            user_ids=user_ids,
            item_ids=item_ids,
            order_ids=order_ids,
            timestamp=timestamps,
            ratings=ratings,
        )
    else:
        np.savez_compressed(
            data_file,
            user_ids=user_ids,
            item_ids=item_ids,
            timestamp=timestamps,
            ratings=ratings,
        )
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
        print(interactions)
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
    data[DEFAULT_FLAG_COL] = "train"
    if random:
        data = sklearn.utils.shuffle(data)
    else:
        data.sort_values(by=[DEFAULT_TIMESTAMP_COL], inplace=True)

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


def data_split(
    data,
    split_type,
    test_rate,
    random=False,
    n_negative=100,
    save_dir=None,
    by_user=False,
    test_copy=10,
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
        random: bool.  Whether random leave one item/basket as testing. only for leave_one_out and leave_one_basket
        test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
        n_negative: Number of negative samples for testing and validation data.
        save_dir: str. Default None. If specified, the split data will be saved to the dir.
        by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
        test_copy: int. Default 10. The number of testing and validation copies.

    Returns:
        Dataframe. The split data. Note that the returned data will not have negative samples.

    """
    if split_type == "random":
        random_split(data, test_rate, by_user)
    elif split_type == "random_basket":
        random_basket_split(data, test_rate, by_user)
    elif split_type == "leave_one_out":
        leave_one_out(data, random)
    elif split_type == "leave_one_basket":
        leave_one_basket(data, random)
    elif split_type == "temporal":
        data = temporal_split(data, test_rate, by_user)
    elif split_type == "temporal_basket":
        data = temporal_basket_split(data, test_rate, by_user)
    else:
        print("[ERROR] wrong split_type.")
        return None
    if by_user:
        split_type = split_type + "_by_user"
    tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
    tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
    tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
    if save_dir is None:
        return data
    save_data(tp_train, save_dir, split_type, "train.npz")
    item_sampler = UnigramTable(data[DEFAULT_ITEM_COL].value_counts().to_dict())
    for i in range(test_copy):
        tp_validate_new = feed_neg_sample(tp_validate, n_negative, item_sampler)
        tp_test_new = feed_neg_sample(tp_test, n_negative, item_sampler)
        save_data(tp_validate_new, save_dir, split_type, "valid_" + str(i) + ".npz")
        save_data(tp_test_new, save_dir, split_type, "test_" + str(i) + ".npz")
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
