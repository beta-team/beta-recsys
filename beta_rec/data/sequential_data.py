import pandas as pd

from ..data.base_data import BaseData
from ..utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

pd.options.mode.chained_assignment = None  # default='warn'


class SequentialData(BaseData):
    r"""A Sequential Data object, which models training input as a squential interactions.

    Re-index all the users and items from raw dataset.

    Args:
        split_dataset (train,valid,test): the split dataset, a tuple consisting of training (DataFrame),
            validate/list of validate (DataFrame), testing/list of testing (DataFrame).
        intersect (bool, optional): remove users and items of test/valid sets that do not exist in the train set.
            If the model is able to predict for new users and new items, this can be :obj:`False`
            (default: :obj:`True`).
        binarize (bool, optional): binarize the rating column of train set 0 or 1, i.e. implicit feedback.
            (default: :obj:`True`).
        bin_thld (int, optional):  the threshold of binarization (default: :obj:`0`).
        normalize (bool, optional): normalize the rating column of train set into [0, 1], i.e. explicit feedback.
            (default: :obj:`False`).
    """

    def __init__(
        self,
        split_dataset,
        config=None,
        intersect=True,
        binarize=True,
        bin_thld=0.0,
        normalize=False,
    ):
        """Initialize GroceryData Class."""
        BaseData.__init__(
            self,
            split_dataset=split_dataset,
            intersect=intersect,
            binarize=binarize,
            bin_thld=bin_thld,
            normalize=normalize,
        )
        self.config = config

    def get_train_seq(self, dump=True, load_save=False):
        """Sample triples or load triples samples from files.

        This method is only applicable for basket based Recommender.

        Args:
            dump (bool, optional): [description]. Defaults to True.
            load_save (bool, optional): [description]. Defaults to False.

        Returns:
            [DataFrame]: [description]
        """
        self.train.sort_values(
            by=[DEFAULT_TIMESTAMP_COL], ascending=False, inplace=True
        )
        train_seq_df = self.train.groupby([DEFAULT_USER_COL])[DEFAULT_ITEM_COL].apply(
            list
        )
        return train_seq_df
