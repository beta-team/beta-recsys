import sys

from beta_rec.datasets.yoochoose import YooChoose

sys.path.append("../")

if __name__ == "__main__":
    dataset = YooChoose()
    dataset.preprocess()
    # interactions = dataset.load_interaction()
    # print(interactions.head())
    # from beta_rec.datasets.data_split import filter_user_item_order
    #
    # # use filter_user_item_order() to make the dataset smaller, just for test
    # interactions = filter_user_item_order(interactions, 5, 5, 5)
    #
    # # use n_test=1 to save time as well
    # dataset.make_leave_one_out(n_test=1)
    dataset.make_random_split(n_test=1)

    #
    # dataset.load_leave_one_out(n_test=1)
    # dataset.load_random_split(n_test=1)
    # train_data, valid_data_li, test_data_li = dataset.load_temporal_split(n_test=1)
    # print(valid_data_li[0]["col_rating"].unique())
    # train_data, valid_data_li, test_data_li = dataset.load_random_split(n_test=1)
    # print(valid_data_li[0]["col_rating"].unique())
