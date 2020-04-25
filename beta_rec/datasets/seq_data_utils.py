import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


def reindex_items(train_data, valid_data=None, test_data=None):
    train_data = train_data.sort_values(by=['col_user', 'col_timestamp'])
    test_data = test_data.sort_values(by=['col_user', 'col_timestamp'])

    # train data
    item_ids = train_data.col_item.unique()
    n_items = len(item_ids)
    item2idx = pd.Series(data=np.arange(len(item_ids))+1,index=item_ids)
    
    # Build itemmap is a DataFrame that have 2 columns (col_item, item_idx)
    itemmap = pd.DataFrame({"col_item": item_ids,
                            'item_idx': item2idx[item_ids].values})
    train_data = pd.merge(train_data, itemmap, on="col_item", how='inner')

    train_data.col_item = train_data.item_idx
    train_data = train_data.drop(columns=['item_idx'])

    train_data = train_data.sort_values(by=['col_user', 'col_timestamp'])

    # test data
    test_data = pd.merge(test_data, itemmap, on="col_item", how='inner')
    test_data.col_item = test_data.item_idx
    test_data = test_data.drop(columns=['item_idx'])
    test_data = test_data.sort_values(by=['col_user', 'col_timestamp'])
    
    # valid data
    if valid_data is not None:
        valid_data = pd.merge(valid_data, itemmap, on="col_item", how='inner')
        valid_data.col_item = valid_data.item_idx
        valid_data = valid_data.drop(columns=['item_idx'])
        valid_data = valid_data.sort_values(by=['col_user', 'col_timestamp'])

    return train_data, valid_data, test_data


def create_seq_db(data):
    """
    Convert interactions of a user to a sequence.
    
    :param data: the dataset to be transformed
    """
    # group by user id and concat item id
    groups = data.groupby('col_user')

    # convert item ids to int, then aggregate them to lists
    aggregated = groups.col_item.agg(col_sequence= lambda x: list(map(int, x)))
    
    result = aggregated
    result.reset_index(inplace=True)
    return result


def dataset_to_seq_target_format(data):
    """
    Convert a list of sequences to (seq,target) format.
    
    :param data: the dataset to be transformed
    """

    iseqs = data['col_sequence']

    out_seqs = []
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            ids += [id]
    return (out_seqs, labs)

class SeqDataset(Dataset):
    """
    define the pytorch Dataset class for sequential datasets.
    """
    def __init__(self, data,print_info=True):
        self.data = data
        if print_info:
            print('-'*80)
            print('Dataset info:')
            print('Number of sessions: {}'.format(len(data[0])))
            print('-'*80)
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])


def collate_fn(data):
    """
    This function will be used to pad the sessions to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each session (before padding)
       It will be used in the Dataloader
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sesss = torch.zeros(len(data), max(lens)).long()
    for i, (sess, label) in enumerate(data):
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
        labels.append(label)
    
    padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, torch.tensor(labels).long(), lens