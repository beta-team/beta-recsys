import sys

from beta_rec.datasets.dunnhumby import Dunnhumby

sys.path.append("../")

if __name__ == "__main__":
    dataset = Dunnhumby()
    dataset.preprocess()
    interactions = dataset.load_interaction()
    from beta_rec.datasets.data_split import filter_user_item_order

    # use filter_user_item_order() to make the dataset smaller, just for test
    interactions = filter_user_item_order(interactions, 100, 100, 100)

    # use n_test=1 to save time as well
    dataset.make_leave_one_basket(interactions, n_test=1)
    dataset.make_leave_one_out(interactions, n_test=1)
    dataset.make_random_basket_split(interactions, n_test=1)
    dataset.make_random_split(interactions, n_test=1)
    dataset.make_temporal_basket_split(interactions, n_test=1)
    dataset.make_temporal_split(interactions, n_test=1)

    dataset.load_leave_one_basket(n_test=1)
    dataset.load_leave_one_out(n_test=1)
    dataset.load_random_basket_split(n_test=1)
    dataset.load_random_split(n_test=1)
    dataset.load_temporal_basket_split(n_test=1)
    train_data, valid_data_li, test_data_li = dataset.load_temporal_split(n_test=1)
