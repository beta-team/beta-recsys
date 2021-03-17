import os

import pandas as pd

from ..data.auxiliary_data import Auxiliary
from ..data.base_data import BaseData
from ..utils.common_util import ensureDir
from ..utils.triple_sampler import Sampler

pd.options.mode.chained_assignment = None  # default='warn'


class GroceryData(BaseData, Auxiliary):
    r"""A Grocery Data object, which consist one more order/basket column than the BaseData.

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
        Auxiliary.__init__(
            self, config=config, n_users=self.n_users, n_items=self.n_items
        )

    def sample_triple_time(self, dump=True, load_save=False):
        """Sample triples or load triples samples from files.

        This method is only applicable for basket based Recommender.

        Returns:
            None

        """
        sample_file_name = (
            "triple_"
            + self.config["dataset"]["dataset"]
            + (
                ("_" + str(self.config["dataset"]["percent"] * 100))
                if "percent" in self.config
                else ""
            )
            + (
                ("_" + str(self.config["model"]["time_step"]))
                if "time_step" in self.config
                else "_10"
            )
            + "_"
            + str(self.config["model"]["n_sample"])
            if "percent" in self.config
            else "" + ".csv"
        )
        self.process_path = self.config["system"]["process_dir"]
        ensureDir(self.process_path)
        sample_file = os.path.join(self.process_path, sample_file_name)
        my_sampler = Sampler(
            self.train,
            sample_file,
            self.config["model"]["n_sample"],
            dump=dump,
            load_save=load_save,
        )
        return my_sampler.sample_by_time(self.config["model"]["time_step"])

    def sample_triple(self, dump=True, load_save=False):
        """Sample triples or load triples samples from files.

        This method is only applicable for basket based Recommender.

        Returns:
            None

        """
        sample_file_name = (
            "triple_"
            + self.config["dataset"]["dataset"]
            + (
                ("_" + str(self.config["dataset"]["percent"] * 100))
                if "percent" in self.config
                else ""
            )
            + "_"
            + str(self.config["model"]["n_sample"])
            if "percent" in self.config
            else "" + ".csv"
        )
        self.process_path = self.config["system"]["process_dir"]
        ensureDir(self.process_path)
        sample_file = os.path.join(self.process_path, sample_file_name)
        my_sampler = Sampler(
            self.train,
            sample_file,
            self.config["model"]["n_sample"],
            dump=dump,
            load_save=load_save,
        )
        return my_sampler.sample()
