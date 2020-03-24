import numpy as np
from datasets import dunnhumby, tafeng, movielens


## to do
def load_user_fea_dic(config, fea_type):
    pass

def load_item_fea_dic(config, fea_type):
    """
    Load item_feature_one.csv item_feature_w2v.csv item_feature_bert.csv

    Returns
    -------
    item_feature dict.

    """
    data_str = config["dataset"]
    root_dir = config["root_dir"]
    print("load basic item featrue for dataset:", data_str, " type:", fea_type)

    if fea_type == "word2vec":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_w2v.csv", "r"
        )
    elif fea_type == "cate":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_cate.csv", "r"
        )
    elif fea_type == "one_hot":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_one.csv", "r"
        )
    elif fea_type == "bert":
        item_feature_file = open(
            root_dir + "datasets/" + data_str + "/raw/item_feature_bert.csv", "r"
        )
    else:
        print(
            "[ERROR]: CANNOT support other feature type, use 'random' user featrue instead!"
        )

    item_feature = {}
    lines = item_feature_file.readlines()
    for index in range(1, len(lines)):
        key_value = lines[index].split(",")
        item_id = int(key_value[0])
        feature = np.array(key_value[1].split(" "), dtype=np.float)
        item_feature[item_id] = feature
    return item_feature


def load_split_dataset(config):
    """
    Loading dataset
    """
    root_dir = config["root_dir"]
    if "test_percent" not in config:
        test_percent = None
    else:
        test_percent = config["test_percent"]
    if config["dataset"] == "dunnhumby":
        if config["data_split"] == "temporal":
            train_df, validate_df, test_df = dunnhumby.load_temporal(root_dir=root_dir)
        elif config["data_split"] == "leave_one_item":
            train_df, validate_df, test_df = dunnhumby.load_leave_one_item(
                root_dir=root_dir
            )
        elif config["data_split"] == "leave_one_basket":
            train_df, validate_df, test_df = dunnhumby.load_leave_one_basket(
                root_dir=root_dir
            )
        else:
            train_df, validate_df, test_df = dunnhumby.load_leave_one_out(
                root_dir=root_dir
            )
    elif config["dataset"] == "tafeng":
        if config["data_split"] == "temporal":
            train_df, validate_df, test_df = tafeng.load_temporal(
                root_dir=root_dir, test_percent=test_percent
            )
        elif config["data_split"] == "leave_one_item":
            train_df, validate_df, test_df = tafeng.load_leave_one_item(
                root_dir=root_dir
            )
        elif config["data_split"] == "leave_one_basket":
            train_df, validate_df, test_df = tafeng.load_leave_one_basket(
                root_dir=root_dir
            )
        else:
            train_df, validate_df, test_df = tafeng.load_leave_one_out(
                root_dir=root_dir
            )
    elif config["dataset"] == "movielens" or config["dataset"] == "ml-1m":
        if config["data_split"] == "temporal":
            train_df, validate_df, test_df = movielens.load_temporal(root_dir=root_dir)
        elif (
            config["data_split"] == "leave_one_item"
            or config["data_split"] == "leave_one_out"
        ):
            train_df, validate_df, test_df = movielens.load_leave_one_out(
                root_dir=root_dir
            )
    else:
        print("get the wrong dataset or data_split.")

    return train_df, validate_df, test_df