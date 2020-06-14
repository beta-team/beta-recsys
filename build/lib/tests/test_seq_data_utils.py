import sys

from torch.utils.data import DataLoader

from beta_rec.datasets.movielens import Movielens_100k
from beta_rec.datasets.seq_data_utils import (
    SeqDataset,
    collate_fn,
    create_seq_db,
    dataset_to_seq_target_format,
    reindex_items,
)

sys.path.append("../")


if __name__ == "__main__":
    ml = Movielens_100k()
    ml.download()
    ml.load_interaction()

    tem_data = ml.make_temporal_split()

    tem_train_data = tem_data[tem_data.col_flag == "train"]
    tem_valid_data = tem_data[tem_data.col_flag == "validate"]
    tem_test_data = tem_data[tem_data.col_flag == "test"]

    # reindex items from 1
    train_data, valid_data, test_data = reindex_items(
        tem_train_data, tem_valid_data, tem_test_data
    )

    # convert interactions to sequences
    seq_train_data = create_seq_db(train_data)

    # convert sequences to (seq, target) format
    load_train_data = dataset_to_seq_target_format(seq_train_data)

    # define pytorch Dataset class for sequential datasets
    load_train_data = SeqDataset(load_train_data)

    # pad the sequences with 0
    load_train_data = DataLoader(
        load_train_data, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
