import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..datasets.dunnhumby import Dunnhumby
from ..datasets.epinions import Epinions
from ..datasets.instacart import Instacart, Instacart_25
from ..datasets.last_fm import LastFM
from ..datasets.movielens import Movielens_1m, Movielens_25m, Movielens_100k
from ..datasets.tafeng import Tafeng


def load_dataset(config):
    """Load datasets.

    Args:
        config (dict): Dictionary of configuration.

    Returns:
        dataset (pandas.DataFrame): Full dataset.
    """
    dataset_mapping = {
        "ml_100k": Movielens_100k,
        "ml_1m": Movielens_1m,
        "ml_25m": Movielens_25m,
        "last_fm": LastFM,
        "tafeng": Tafeng,
        "epinions": Epinions,
        "dunnhumby": Dunnhumby,
        "instacart": Instacart,
        "instacart_25": Instacart_25,
    }
    dataset = dataset_mapping[config["dataset"]["dataset"]]()
    return dataset


def reindex_items(train_data, valid_data=None, test_data=None):
    """Reindex the item ids.

    Item ids are reindexed from 1. "0" is left for padding.

    Args:
        train_data (pandas.DataFrame): Training set.
        valid_data (pandas.DataFrame): Validation set.
        test_data (pandas.DataFrame): Test set.

    Returns:
        train_data (pandas.DataFrame): Reindexed training set.
        valid_data (pandas.DataFrame): Reindexed validation set.
        test_data (pandas.DataFrame): Reindexed test set.
    """
    train_data = train_data.sort_values(by=["col_user", "col_timestamp"])
    test_data = test_data.sort_values(by=["col_user", "col_timestamp"])

    # train data
    item_ids = train_data.col_item.unique()
    item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)

    # Build itemmap is a DataFrame that have 2 columns (col_item, item_idx)
    itemmap = pd.DataFrame(
        {"col_item": item_ids, "item_idx": item2idx[item_ids].values}
    )
    train_data = pd.merge(train_data, itemmap, on="col_item", how="inner")

    train_data.col_item = train_data.item_idx
    train_data = train_data.drop(columns=["item_idx"])

    train_data = train_data.sort_values(by=["col_user", "col_timestamp"])

    # test data
    test_data = pd.merge(test_data, itemmap, on="col_item", how="inner")
    test_data.col_item = test_data.item_idx
    test_data = test_data.drop(columns=["item_idx"])
    test_data = test_data.sort_values(by=["col_user", "col_timestamp"])

    # valid data
    if valid_data is not None:
        valid_data = pd.merge(valid_data, itemmap, on="col_item", how="inner")
        valid_data.col_item = valid_data.item_idx
        valid_data = valid_data.drop(columns=["item_idx"])
        valid_data = valid_data.sort_values(by=["col_user", "col_timestamp"])

    return train_data, valid_data, test_data


def create_seq_db(data):
    """Convert interactions of a user to a sequence.

    Args:
        data (pandas.DataFrame): The dataset to be transformed.

    Returns:
        result (pandas.DataFrame): Transformed dataset with "col_user" and "col_sequence".
    """
    # group by user id and concat item id
    groups = data.groupby("col_user")

    # convert item ids to int, then aggregate them to lists
    aggregated = groups.col_item.agg(col_sequence=lambda x: list(map(int, x)))

    result = aggregated
    result.reset_index(inplace=True)
    return result


def dataset_to_seq_target_format(data):
    """Convert a list of sequences to (seq,target) format.

    Args:
        data (pandas.DataFrame): The dataset to be transformed.

    Returns:
        out_seqs (List): Context sequence.
        labs (List): Labels of the context sequence, each element is the last item in the origin sequence.
    """
    iseqs = data["col_sequence"]

    out_seqs = []
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            ids += [id]
    return out_seqs, labs


class SeqDataset(Dataset):
    """Sequential Dataset."""

    def __init__(self, data, print_info=True):
        """Init SeqDataset Class."""
        self.data = data
        if print_info:
            print("-" * 80)
            print("Dataset info:")
            print("Number of sessions: {}".format(len(data[0])))
            print("-" * 80)

    def __getitem__(self, index):
        """Get an item from the dataset by index."""
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.data[0])


def collate_fn(data):
    """Pad the sequences.

    This function will be used to pad the sessions to max length
    in the batch and transpose the batch from
    batch_size x max_seq_len to max_seq_len x batch_size.
    It will return padded vectors, labels and lengths of each session (before padding)
    It will be used in the Dataloader.

    Args:
        data (pytorch Dataset): Sequential dataset.

    Returns:
        padded_sesss (Tensor): Padded vectors.
        labels (Tensor): Target item.
        lens (list): Lengths of each padded vector.
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sesss = torch.zeros(len(data), max(lens)).long()
    for i, (sess, label) in enumerate(data):
        padded_sesss[i, : lens[i]] = torch.LongTensor(sess)
        labels.append(label)
    padded_sesss = padded_sesss.transpose(0, 1)
    return padded_sesss, torch.tensor(labels).long(), lens
