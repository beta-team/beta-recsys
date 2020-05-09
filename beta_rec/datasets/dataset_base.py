import os
import shutil
import pandas as pd
from tabulate import tabulate
from py7zr import unpack_7zarchive
from beta_rec.utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ORDER_COL
from beta_rec.utils.common_util import (
    get_dataframe_from_npz,
    save_dataframe_as_npz,
    timeit,
    un_zip,
)
from beta_rec.utils.download import download_file, get_format
from beta_rec.utils.onedrive import OneDrive
from beta_rec.datasets.data_split import (
    split_data,
    generate_parameterized_path,
    load_split_data,
    filter_user_item_order,
    filter_user_item,
)

default_root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

# register 7z unpack
shutil.register_unpack_format("7zip", ['.7z'], unpack_7zarchive)


class DatasetBase(object):
    def __init__(
        self,
        dataset_name,
        url=None,
        root_dir=None,
        manual_download_url=None,
        processed_leave_one_out_url="",
        processed_leave_one_basket_url="",
        processed_random_split_url="",
        processed_random_basket_split_url="",
        processed_temporal_split_url="",
        processed_temporal_basket_split_url="",
        tips=None,
    ):
        """Dataset base that any other datasets need to inherit from

        This is an beta dataset which can derive to other dataset.
        Several directory that store the dataset file would be created in the initial process.

        Args:
            dataset_name: the dataset name that a folder can be created with.
            url: the url that can be downloaded the dataset file.
            manual_download_url: the url that users need to download manually
        """
        self.url = url
        self.manual_download_url = manual_download_url if manual_download_url else url

        self.processed_leave_one_out_url = processed_leave_one_out_url
        self.processed_leave_one_basket_url = processed_leave_one_basket_url
        self.processed_random_split_url = processed_random_split_url
        self.processed_random_basket_split_url = processed_random_basket_split_url
        self.processed_temporal_split_url = processed_temporal_split_url
        self.processed_temporal_basket_split_url = processed_temporal_basket_split_url

        if tips is None:
            tips = f"please download the dataset by your self via {self.manual_download_url}, rename to " + \
                   f"{self.dataset_name} and put it into {self.raw_path} after decompression "
        self.tips = tips

        self.dataset_name = dataset_name
        # compatible method for the previous version
        self.save_dataframe_as_npz = save_dataframe_as_npz
        # create the root datasets directory
        if not root_dir:
            root_dir = default_root_dir
        self.dataset_dir = os.path.join(root_dir, "datasets")
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        # create the dataset directory
        self.dataset_dir = os.path.join(self.dataset_dir, dataset_name)
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        # create the raw directory
        self.raw_path = os.path.join(self.dataset_dir, "raw")
        if not os.path.exists(self.raw_path):
            os.mkdir(self.raw_path)

        self.processed_path = os.path.join(self.dataset_dir, "processed")
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)

        if not url:
            print(self.tips)

    @timeit
    def download(self):
        """Download the raw dataset.

        Download the dataset with the given url and unpack the file.
        """
        if not self.url:
            raise RuntimeError(self.tips)

        download_file_name = os.path.join(
            self.raw_path, os.path.splitext(os.path.basename(self.url))[0]
        )
        file_format = self.url.split(".")[-1]
        raw_file_path = os.path.join(
            self.raw_path, f"{self.dataset_name}.{file_format}"
        )

        if not os.path.exists(raw_file_path):
            download_file(self.url, raw_file_path)
            shutil.unpack_archive(
                raw_file_path, self.raw_path, format=get_format(file_format)
            )
            if not os.path.exists(download_file_name):
                return
            elif os.path.isdir(download_file_name):
                os.rename(
                    download_file_name, os.path.join(self.raw_path, self.dataset_name)
                )
            else:
                os.rename(
                    download_file_name,
                    os.path.join(
                        self.raw_path,
                        f'{self.dataset_name}.{download_file_name.split(".")[-1]}',
                    ),
                )

    @timeit
    def preprocess(self):
        """Preprocess the raw file.

        A virtual function that needs to be implement in the derived class.
        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        raise RuntimeError(f"please implement this function!")

    def load_interaction(self):
        """Load the user-item interaction

        Load the interaction from the processed file(Need to preprocess the raw file before loading)
        """
        processed_file_path = os.path.join(
            self.processed_path, f"{self.dataset_name}_interaction.npz"
        )
        if not os.path.exists(os.path.join(processed_file_path)):
            self.preprocess()
        data = get_dataframe_from_npz(processed_file_path)
        print("-" * 80)
        print("raw interaction statistics")
        print(
            tabulate(
                data.agg(["count", "nunique"]),
                headers=data.columns,
                tablefmt="psql",
                disable_numparse=True,
            )
        )
        print("-" * 80)
        return data

    @timeit
    def make_leave_one_out(self, data=None, random=False, n_negative=100, n_test=10):
        """generate split data with leave_one_out.

        Generate split data with leave_one_out method.

        Args:
            data (DataFrame): DataFrame to be split.
                - Default is None. It will load the raw interaction, with a default filter
                    '''
                    filter_user_item(data, min_u_c=0, min_i_c=3)
                    '''
                - Users can specify their filtered data by using filter methods in data_split.py
            random: bool. Whether randomly leave one item as testing.
            n_negative:  Number of negative samples for testing and validation data.
            n_test: int. Default 10. The number of testing and validation copies.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """
        if data is None:
            data = self.load_interaction()
            data = filter_user_item(data, min_u_c=0, min_i_c=3)

        if not isinstance(data, pd.DataFrame):
            raise RuntimeError("data is not a type of DataFrame")

        if DEFAULT_TIMESTAMP_COL not in data.columns:
            random = True

        result = split_data(
            data,
            split_type="leave_one_out",
            test_rate=0,
            random=random,
            n_negative=n_negative,
            save_dir=self.processed_path,
            n_test=n_test,
        )
        return result

    @timeit
    def make_leave_one_basket(self, data=None, random=False, n_negative=100, n_test=10):
        """generate split data with leave_one_basket.

        Generate split data with leave_one_basket method.

        Args:
            data (DataFrame): DataFrame to be split.
                - Default is None. It will load the raw interaction, with a default filter
                    '''
                    filter_user_item(data, min_u_c=0, min_o_c=3, min_i_c=0)
                    '''
                - Users can specify their filtered data by using filter methods in data_split.py
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            n_test: int. Default 10. The number of testing and validation copies.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """
        if data is None:
            data = self.load_interaction()
            data = filter_user_item_order(data, min_u_c=0, min_o_c=3, min_i_c=0)

        if not isinstance(data, pd.DataFrame):
            raise RuntimeError("data is not a type of DataFrame")

        if DEFAULT_TIMESTAMP_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an TIMESTAMP_COL")

        if DEFAULT_ORDER_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an ORDER_COL")

        result = split_data(
            data,
            split_type="leave_one_basket",
            test_rate=0,
            random=random,
            n_negative=n_negative,
            save_dir=self.processed_path,
            n_test=n_test,
        )
        return result

    @timeit
    def make_random_split(
        self, data=None, test_rate=0.1, n_negative=100, by_user=False, n_test=10
    ):
        """generate split data with random_split.

        Generate split data with random_split method

        Args:
            data (DataFrame): DataFrame to be split.
                - Default is None. It will load the raw interaction, with a default filter
                ```
                data = filter_user_item(data, min_u_c=10, min_i_c=10)
                ```
                - Users can specify their filtered data by using filter methods in data_split.py
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """
        if data is None:
            data = self.load_interaction()
            data = filter_user_item(data, min_u_c=10, min_i_c=10)

        if not isinstance(data, pd.DataFrame):
            raise RuntimeError("data is not a type of DataFrame")

        result = split_data(
            data,
            split_type="random",
            test_rate=test_rate,
            n_negative=n_negative,
            save_dir=self.processed_path,
            by_user=by_user,
            n_test=n_test,
        )
        return result

    @timeit
    def make_random_basket_split(
        self, data=None, test_rate=0.1, n_negative=100, by_user=False, n_test=10
    ):
        """generate split data with random_basket_split.

        Generate split data with random_basket_split method.

        Args:
            data (DataFrame): DataFrame to be split.
                - Default is None. It will load the raw interaction, with a default filter
                ```
                data = filter_user_item_order(data, min_u_c=10, min_o_c=10, min_i_c=10)
                ```
                - Users can specify their filtered data by using filter methods in data_split.py
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """
        if data is None:
            data = self.load_interaction()
            data = filter_user_item_order(data, min_u_c=10, min_o_c=10, min_i_c=10)

        if not isinstance(data, pd.DataFrame):
            raise RuntimeError("data is not a type of DataFrame")

        if DEFAULT_ORDER_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an ORDER_COL")

        result = split_data(
            data,
            split_type="random_basket",
            test_rate=test_rate,
            n_negative=n_negative,
            save_dir=self.processed_path,
            by_user=by_user,
            n_test=n_test,
        )
        return result

    @timeit
    def make_temporal_split(
        self, data=None, test_rate=0.1, n_negative=100, by_user=False, n_test=10
    ):
        """generate split data with temporal_split.

        Generate split data with temporal_split method.

        Args:
            data (DataFrame): DataFrame to be split.
                - Default is None. It will load the raw interaction, with a default filter
                ```
                data = filter_user_item(data, min_u_c=10, min_i_c=10)
                ```
                - Users can specify their filtered data by using filter methods in data_split.py
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """
        if data is None:
            data = self.load_interaction()
            data = filter_user_item(data, min_u_c=10, min_i_c=10)

        if not isinstance(data, pd.DataFrame):
            raise RuntimeError("data is not a type of DataFrame")

        if DEFAULT_TIMESTAMP_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an TIMESTAMP_COL")

        result = split_data(
            data,
            split_type="temporal",
            test_rate=test_rate,
            n_negative=n_negative,
            save_dir=self.processed_path,
            by_user=by_user,
            n_test=n_test,
        )
        return result

    @timeit
    def make_temporal_basket_split(
        self, data=None, test_rate=0.1, n_negative=100, by_user=False, n_test=10
    ):
        """generate split data with temporal_basket_split.

        Generate split data with temporal_basket_split method.

        Args:
            data (DataFrame): DataFrame to be split.
                - Default is None. It will load the raw interaction, with a default filter
                ```
                data = filter_user_item_order(data, min_u_c=10, min_o_c=10, min_i_c=10)
                ```
                - Users can specify their filtered data by using filter methods in data_split.py
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """

        if data is None:
            data = self.load_interaction()
            data = filter_user_item_order(data, min_u_c=10, min_o_c=10, min_i_c=10)

        if not isinstance(data, pd.DataFrame):
            raise RuntimeError("data is not a type of DataFrame")

        if DEFAULT_TIMESTAMP_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an TIMESTAMP_COL")

        if DEFAULT_ORDER_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an ORDER_COL")

        result = split_data(
            data,
            split_type="temporal_basket",
            test_rate=test_rate,
            n_negative=n_negative,
            save_dir=self.processed_path,
            by_user=by_user,
            n_test=n_test,
        )
        return result

    def load_leave_one_out(self, random=False, n_negative=100, n_test=10):
        """load split data generated by leave_out_out without random select.

        Load split data generated by leave_out_out without random select from Onedrive.

        Args:
            random: bool. Whether randomly leave one item as testing.
            n_negative:  Number of negative samples for testing and validation data.
            n_test: int. Default 10. The number of testing and validation copies.
                    If n_test==0, will load the original (no negative items) valid and test datasets.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """

        processed_leave_one_out_path = os.path.join(
            self.processed_path, "leave_one_out"
        )
        if not os.path.exists(processed_leave_one_out_path):
            os.mkdir(processed_leave_one_out_path)

        parameterized_path = generate_parameterized_path(
            test_rate=0, random=random, n_negative=n_negative, by_user=False
        )

        download_path = processed_leave_one_out_path
        processed_leave_one_out_path = os.path.join(
            processed_leave_one_out_path, parameterized_path
        )

        if not os.path.exists(processed_leave_one_out_path):
            if random is False and n_negative == 100:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(
                    url=self.processed_leave_one_out_url, path=download_path
                )
                folder.download()
                un_zip(processed_leave_one_out_path + ".zip", download_path)
            else:
                # make
                self.make_leave_one_out(
                    random=random, n_negative=n_negative, n_test=n_test
                )

        # load data from local storage
        return load_split_data(processed_leave_one_out_path, n_test=n_test)

    def load_leave_one_basket(self, random=False, n_negative=100, n_test=10):
        """load split date generated by leave_one_basket without random select.

        Load split data generated by leave_one_basket without random select from Onedrive.

        Args:
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            n_test: int. Default 10. The number of testing and validation copies.
                    If n_test==0, will load the original (no negative items) valid and test datasets.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """

        processed_leave_one_basket_path = os.path.join(
            self.processed_path, "leave_one_basket"
        )
        if not os.path.exists(processed_leave_one_basket_path):
            os.mkdir(processed_leave_one_basket_path)

        parameterized_path = generate_parameterized_path(
            test_rate=0, random=random, n_negative=n_negative, by_user=False
        )
        download_path = processed_leave_one_basket_path
        processed_leave_one_basket_path = os.path.join(
            processed_leave_one_basket_path, parameterized_path
        )
        if not os.path.exists(processed_leave_one_basket_path):
            if random is False and n_negative == 100:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(
                    url=self.processed_leave_one_basket_url, path=download_path
                )
                folder.download()
                un_zip(processed_leave_one_basket_path + ".zip", download_path)
            else:
                # make
                self.make_leave_one_basket(
                    random=random, n_negative=n_negative, n_test=n_test
                )

        # load data from local storage
        return load_split_data(processed_leave_one_basket_path, n_test=n_test)

    def load_random_split(
        self, test_rate=0.1, random=False, n_negative=100, by_user=False, n_test=10
    ):
        """load split date generated by random_split.

        Load split data generated by random_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as
                        test data.
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.
                    If n_test==0, will load the original (no negative items) valid and test datasets.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """

        processed_random_split_path = os.path.join(self.processed_path, "random")
        if not os.path.exists(processed_random_split_path):
            os.mkdir(processed_random_split_path)

        parameterized_path = generate_parameterized_path(
            test_rate=test_rate, random=random, n_negative=n_negative, by_user=by_user
        )
        download_path = processed_random_split_path
        processed_random_split_path = os.path.join(
            processed_random_split_path, parameterized_path
        )
        if not os.path.exists(processed_random_split_path):
            if (
                test_rate == 0.1
                and random is False
                and n_negative == 100
                and by_user is False
            ):
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(
                    url=self.processed_random_split_url, path=download_path
                )
                folder.download()
                un_zip(processed_random_split_path + ".zip", download_path)
            else:
                # make
                self.make_random_split(
                    test_rate=test_rate,
                    random=random,
                    n_negative=n_negative,
                    by_user=by_user,
                    n_test=n_test,
                )

        # load data from local storage
        return load_split_data(processed_random_split_path, n_test=n_test)

    def load_random_basket_split(
        self, test_rate=0.1, random=False, n_negative=100, by_user=False, n_test=10
    ):
        """load split date generated by random_basket_split.

        Load split data generated by random_basket_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as
                        test data.
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.
                    If n_test==0, will load the original (no negative items) valid and test datasets.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """

        processed_random_basket_split_path = os.path.join(
            self.processed_path, "random_basket"
        )
        if not os.path.exists(processed_random_basket_split_path):
            os.mkdir(processed_random_basket_split_path)

        parameterized_path = generate_parameterized_path(
            test_rate=test_rate, random=random, n_negative=n_negative, by_user=by_user
        )

        download_path = processed_random_basket_split_path
        processed_random_basket_split_path = os.path.join(
            processed_random_basket_split_path, parameterized_path
        )
        if not os.path.exists(processed_random_basket_split_path):
            if (
                test_rate == 0.1
                and random is False
                and n_negative == 100
                and by_user is False
            ):
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(
                    url=self.processed_random_basket_split_url, path=download_path
                )
                folder.download()
                un_zip(processed_random_basket_split_path + ".zip", download_path)
            else:
                # make
                self.make_random_basket_split(
                    test_rate=test_rate,
                    random=random,
                    n_negative=n_negative,
                    by_user=by_user,
                    n_test=n_test,
                )

        # load data from local storage
        return load_split_data(processed_random_basket_split_path, n_test=n_test)

    def load_temporal_split(
        self, test_rate=0.1, n_negative=100, by_user=False, n_test=10
    ):
        """load split date generated by temporal_split.

        Load split data generated by temporal_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as
                        test data.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.
                    If n_test==0, will load the original (no negative items) valid and test datasets.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """

        processed_temporal_split_path = os.path.join(self.processed_path, "temporal")

        if not os.path.exists(processed_temporal_split_path):
            os.mkdir(processed_temporal_split_path)

        parameterized_path = generate_parameterized_path(
            test_rate=test_rate, random=False, n_negative=n_negative, by_user=by_user
        )

        download_path = processed_temporal_split_path
        processed_temporal_split_path = os.path.join(
            processed_temporal_split_path, parameterized_path
        )
        if not os.path.exists(processed_temporal_split_path):
            if test_rate == 0.1 and n_negative == 100 and by_user is False:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(
                    url=self.processed_temporal_split_url, path=download_path
                )
                folder.download()
                un_zip(processed_temporal_split_path + ".zip", download_path)
            else:
                # make
                self.make_temporal_split(
                    test_rate=test_rate,
                    n_negative=n_negative,
                    by_user=by_user,
                    n_test=n_test,
                )

        # load data from local storage
        return load_split_data(processed_temporal_split_path, n_test=n_test)

    def load_temporal_basket_split(
        self, test_rate=0.1, n_negative=100, by_user=False, n_test=10
    ):
        """load split date generated by temporal_basket_split.

        Load split data generated by temporal_basket_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as
                        test data.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            n_test: int. Default 10. The number of testing and validation copies.
                    If n_test==0, will load the original (no negative items) valid and test datasets.

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """

        processed_temporal_basket_split_path = os.path.join(
            self.processed_path, "temporal_basket"
        )
        if not os.path.exists(processed_temporal_basket_split_path):
            os.mkdir(processed_temporal_basket_split_path)

        parameterized_path = generate_parameterized_path(
            test_rate=test_rate, random=False, n_negative=n_negative, by_user=by_user
        )

        download_path = processed_temporal_basket_split_path
        processed_temporal_basket_split_path = os.path.join(
            processed_temporal_basket_split_path, parameterized_path
        )
        if not os.path.exists(processed_temporal_basket_split_path):
            if test_rate == 0.1 and n_negative == 100 and by_user is False:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(
                    url=self.processed_temporal_basket_split_url, path=download_path
                )
                folder.download()
                un_zip(processed_temporal_basket_split_path + ".zip", download_path)
            else:
                # make
                self.make_temporal_basket_split(
                    test_rate=test_rate,
                    n_negative=n_negative,
                    by_user=by_user,
                    n_test=n_test,
                )

        # load data from local storage
        return load_split_data(processed_temporal_basket_split_path, n_test=n_test)

    def load_split(self, config):
        """ Load split data by config dict.

        Args:
            config (dict): config (dict): Dictionary of configuration

        Returns:
            train_data (DataFrame): Interaction for training.
            valid_data list(DataFrame): List of interactions for validation
            test_data list(DataFrame): List of interactions for testing
        """
        data_split_str = config["data_split"]

        split_paras = {}
        split_paras["test_rate"] = config["test_rate"] if "test_rate" in config else 0.1
        split_paras["random"] = config["random"] if "random" in config else False
        split_paras["n_negative"] = (
            config["n_negative"] if "n_negative" in config else 100
        )
        split_paras["by_user"] = config["by_user"] if "by_user" in config else False
        split_paras["n_test"] = config["n_test"] if "n_test" in config else 10

        data_split_mapping = {
            "leave_one_out": self.load_leave_one_out,
            "leave_one_basket": self.load_leave_one_basket,
            "random_split": self.load_random_split,
            "random_basket_split": self.load_random_basket_split,
            "temporal": self.load_temporal_split,
            "temporal_basket": self.load_temporal_basket_split,
        }

        split_para_mapping = {
            "leave_one_out": ["random", "n_negative", "n_test"],
            "leave_one_basket": ["random", "n_negative", "n_test"],
            "random_split": ["test_rate", "by_user", "n_negative", "n_test"],
            "random_basket_split": ["test_rate", "by_user", "n_negative", "n_test"],
            "temporal": ["test_rate", "by_user", "n_negative", "n_test"],
            "temporal_basket": ["test_rate", "by_user", "n_negative", "n_test"],
        }
        para_dic = {
            split_para_key: split_paras[split_para_key]
            if split_para_key in split_paras
            else None
            for split_para_key in split_para_mapping[data_split_str]
        }
        train_data, valid_data, test_data = data_split_mapping[data_split_str](
            **para_dic
        )
        return train_data, valid_data, test_data
