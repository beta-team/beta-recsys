import os
import shutil
import pandas as pd
import numpy as np

from beta_rec.utils.constants import *
from beta_rec.utils.download import download_file, get_format
from beta_rec.datasets.data_split import data_split, generate_parameterized_path
from beta_rec.utils.onedrive import OneDrive

class DatasetBase(object):
    def __init__(self, dataset_name, url=None, manual_download_url=None):
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

        self.dataset_name = dataset_name

        # create the root datasets directory
        self.dataset_dir = os.path.join(os.path.abspath('.'), 'datasets')
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        # create the dataset directory
        self.dataset_dir = os.path.join(self.dataset_dir, dataset_name)
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        self.raw_path = os.path.join(self.dataset_dir, 'raw')
        if not os.path.exists(self.raw_path):
            os.mkdir(self.raw_path)

        self.processed_path = os.path.join(self.dataset_dir, 'processed')
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)

        if not url:
            print(
                f'please download the dataset by your self via {self.manual_download_url} and put it into {self.raw_path} after decompression')

    def download(self):
        """Download the raw dataset.

        Download the dataset with the given url and unpack the file.
        """
        if not self.url:
            raise RuntimeError(
                f'please download the dataset by your self via {self.manual_download_url} and put it into {self.raw_path} after decompression')

        download_file_name = os.path.join(self.raw_path, os.path.splitext(os.path.basename(self.url))[0])
        file_format = self.url.split('.')[-1]
        raw_file_path = os.path.join(self.raw_path, f'{self.dataset_name}.{file_format}')

        if not os.path.exists(raw_file_path):
            download_file(self.url, raw_file_path)
            shutil.unpack_archive(raw_file_path, self.raw_path, format=get_format(file_format))
            if not os.path.exists(download_file_name):
                return
            elif os.path.isdir(download_file_name):
                os.rename(download_file_name, os.path.join(self.raw_path, self.dataset_name))
            else:
                os.rename(download_file_name,
                          os.path.join(self.raw_path, f'{self.dataset_name}.{download_file_name.split(".")[-1]}'))

    def preprocess(self):
        """Preprocess the raw file.

        A virtual function that needs to be implement in the dervied class.
        Preprocess the file downloaded via the url,
        convert it to a dataframe consist of the user-item interaction
        and save in the processed directory.
        """
        raise RuntimeError(f'please implement this function!')

    def load_interaction(self):
        """Load the user-item interaction

        Load the interaction from the processed file(Need to preprocess the raw file before loading)
        """
        processed_file_path = os.path.join(self.processed_path, f'{self.dataset_name}_interaction.npz')
        if not os.path.exists(os.path.join(processed_file_path)):
            self.preprocess()
        data = self.get_dataframe_from_npz(processed_file_path)
        return data

    def save_dataframe_as_npz(self, data, data_file):
        """Save Dataframe in compressed format

        Save and convert the Dataframe to npz file.
        """
        user_ids = data[DEFAULT_USER_COL].to_numpy(dtype=np.long)
        item_ids = data[DEFAULT_ITEM_COL].to_numpy(dtype=np.long)
        if DEFAULT_TIMESTAMP_COL in data.columns:
            timestamps = data[DEFAULT_TIMESTAMP_COL].to_numpy(dtype=np.long)
        ratings = data[DEFAULT_RATING_COL].to_numpy(dtype=np.float32)
        if DEFAULT_TIMESTAMP_COL in data.columns:
            np.savez_compressed(
                data_file,
                user_ids=user_ids,
                item_ids=item_ids,
                timestamp=timestamps,
                ratings=ratings,
            )
        else:
            np.savez_compressed(
                data_file,
                user_ids=user_ids,
                item_ids=item_ids,
                ratings=ratings,
            )

    def get_dataframe_from_npz(self, data_file):
        """Get the Dataframe from npz file

        Get the Dataframe from npz file
        """
        np_data = np.load(data_file)
        if "timestamp" in np_data:
            data = pd.DataFrame(
                data={
                    DEFAULT_USER_COL: np_data["user_ids"],
                    DEFAULT_ITEM_COL: np_data["item_ids"],
                    DEFAULT_RATING_COL: np_data["ratings"],
                    DEFAULT_TIMESTAMP_COL: np_data["timestamp"],
                }
            )
        else:
            data = pd.DataFrame(
                data={
                    DEFAULT_USER_COL: np_data["user_ids"],
                    DEFAULT_ITEM_COL: np_data["item_ids"],
                    DEFAULT_RATING_COL: np_data["ratings"],
                }
            )
        return data

    def make_leave_one_out(self, random=False, n_negative=2, test_copy=10):
        """generate split data with leave_one_out.

        Generate split data with leave_one_out method.

        Args:
            random: bool. Whether randomly leave one item as testing.
            n_negative:  Number of negative samples for testing and validation data.
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        data = self.load_interaction()
        result = data_split(data, split_type="leave_one_out", test_rate=0, random=random, n_negative=n_negative,
                            save_dir=self.processed_path, test_copy=test_copy)
        return result

    def make_leave_one_basket(self, random=False, n_negative=2, test_copy=10):
        """generate split data with leave_one_basket.

        Generate split data with leave_one_basket method.

        Args:
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        data = self.load_interaction()
        if DEFAULT_ORDER_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an ORDER_COL")

        result = data_split(data, split_type="leave_one_basket", test_rate=0, random=random, n_negative=n_negative,
                            save_dir=self.processed_path, test_copy=test_copy)
        return result

    def make_random_split(self, test_rate=0.1, random=False, n_negative=2, by_user=False, test_copy=10):
        """generate split data with random_split.

        Generate split data with random_split method

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        data = self.load_interaction()
        result = data_split(data, split_type="random", test_rate=test_rate, random=random, n_negative=n_negative,
                            save_dir=self.processed_path, by_user=by_user, test_copy=test_copy)
        return result

    def make_random_basket_split(self, test_rate=0.1, random=False, n_negative=2, by_user=False, test_copy=10):
        """generate split data with random_basket_split.

        Generate split data with random_basket_split method.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        data = self.load_interaction()
        if DEFAULT_ORDER_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an ORDER_COL")

        result = data_split(data, split_type="random_basket", test_rate=test_rate, random=random,
                            n_negative=n_negative, save_dir=self.processed_path, by_user=by_user, test_copy=test_copy)
        return result

    def make_temporal_split(self, test_rate=0.1, n_negative=2, by_user=False, test_copy=10):
        """generate split data with temporal_split.

        Generate split data with temporal_split method.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        data = self.load_interaction()
        result = data_split(data, split_type="temporal", test_rate=test_rate, n_negative=n_negative,
                            save_dir=self.processed_path, by_user=by_user, test_copy=test_copy)
        return result

    def make_temporal_basket_split(self, test_rate=0.1, n_negative=2, by_user=False, test_copy=10):
        """generate split data with temporal_basket_split.

        Generate split data with temporal_basket_split method.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        data = self.load_interaction()
        if DEFAULT_ORDER_COL not in data.columns:
            raise RuntimeError("This dataset doesn't have an ORDER_COL")

        result = data_split(data, split_type="temporal_basket", test_rate=test_rate, n_negative=n_negative,
                            save_dir=self.processed_path, by_user=by_user, test_copy=test_copy)
        return result

    def load_leave_one_out(self, random=False, n_negative=100, test_copy=10):
        """load split data generated by leave_out_out without random select.

        Load split data generated by leave_out_out without random select from Onedrive.

        Args:
            random: bool. Whether randomly leave one item as testing.
            n_negative:  Number of negative samples for testing and validation data.
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """

        processed_leave_one_out_path = os.path.join(self.processed_path, "leave_one_out")
        if not os.path.exists(processed_leave_one_out_path):
            os.mkdir(processed_leave_one_out_path)

        parameterized_path = generate_parameterized_path(test_rate=0, random=random, n_negative=n_negative,
                                                         by_user=False, test_copy=test_copy)
        processed_leave_one_out_path = os.path.join(processed_leave_one_out_path, parameterized_path)
        if not os.path.exists(processed_leave_one_out_path):
            if random is False and n_negative == 2 and test_copy == 10:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(url=self.processed_leave_one_out_url, path=processed_leave_one_out_path)
                folder.download()
            else:
                # make
                self.make_leave_one_out(random=random, n_negative=n_negative, test_copy=test_copy)

        # load data from local storage
        train_file = os.path.join(processed_leave_one_out_path, "train.npz")
        train_data = self.get_dataframe_from_npz(train_file)

        valid_file = os.path.join(processed_leave_one_out_path, "valid_0.npz")
        valid_data = self.get_dataframe_from_npz(valid_file)

        test_file = os.path.join(processed_leave_one_out_path, "test_0.npz")
        test_data = self.get_dataframe_from_npz(test_file)
        return train_data, valid_data, test_data

    def load_leave_one_basket(self, random=False, n_negative=2, test_copy=10):
        """load split date generated by leave_one_basket without random select.

        Load split data generated by leave_one_basket without random select from Onedrive.

        Args:
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        processed_leave_one_basket_path = os.path.join(self.processed_path, "leave_one_basket")
        if not os.path.exists(processed_leave_one_basket_path):
            os.mkdir(processed_leave_one_basket_path)

        parameterized_path = generate_parameterized_path(test_rate=0, random=random, n_negative=n_negative,
                                                         by_user=False, test_copy=test_copy)
        processed_leave_one_basket_path = os.path.join(processed_leave_one_basket_path, parameterized_path)
        if not os.path.exists(processed_leave_one_basket_path):
            if random is False and n_negative == 2 and test_copy == 10:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(url=self.processed_leave_one_basket_url, path=processed_leave_one_basket_path)
                folder.download()
            else:
                # make
                self.make_leave_one_basket(random=random, n_negative=n_negative, test_copy=test_copy)

        # load data from local storage
        train_file = os.path.join(processed_leave_one_basket_path, "train.npz")
        train_data = self.get_dataframe_from_npz(train_file)

        valid_file = os.path.join(processed_leave_one_basket_path, "valid_0.npz")
        valid_data = self.get_dataframe_from_npz(valid_file)

        test_file = os.path.join(processed_leave_one_basket_path, "test_0.npz")
        test_data = self.get_dataframe_from_npz(test_file)
        return train_data, valid_data, test_data

    def load_random_split(self, test_rate=0.1, random=False, n_negative=2, by_user=False, test_copy=10):
        """load split date generated by random_split.

        Load split data generated by random_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        processed_random_split_path = os.path.join(self.processed_path, "random")
        if not os.path.exists(processed_random_split_path):
            os.mkdir(processed_random_split_path)

        parameterized_path = generate_parameterized_path(test_rate=test_rate, random=random, n_negative=n_negative,
                                                         by_user=by_user, test_copy=test_copy)
        processed_random_split_path = os.path.join(processed_random_split_path, parameterized_path)
        if not os.path.exists(processed_random_split_path):
            if test_rate == 0.1 and random is False and n_negative == 2 and by_user is False and test_copy == 10:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(url=self.processed_random_split_url, path=processed_random_split_path)
                folder.download()
            else:
                # make
                self.make_random_split(test_rate=test_rate, random=random, n_negative=n_negative, by_user=by_user,
                                       test_copy=test_copy)

        # load data from local storage
        train_file = os.path.join(processed_random_split_path, "train.npz")
        train_data = self.get_dataframe_from_npz(train_file)

        valid_file = os.path.join(processed_random_split_path, "valid_0.npz")
        valid_data = self.get_dataframe_from_npz(valid_file)

        test_file = os.path.join(processed_random_split_path, "test_0.npz")
        test_data = self.get_dataframe_from_npz(test_file)
        return train_data, valid_data, test_data

    def load_random_basket_split(self, test_rate=0.1, random=False, n_negative=2, by_user=False, test_copy=10):
        """load split date generated by random_basket_split.

        Load split data generated by random_basket_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            random: bool. Whether randomly leave one basket as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        processed_random_basket_split_path = os.path.join(self.processed_path, "random_basket")
        if not os.path.exists(processed_random_basket_split_path):
            os.mkdir(processed_random_basket_split_path)

        parameterized_path = generate_parameterized_path(test_rate=test_rate, random=random, n_negative=n_negative,
                                                         by_user=by_user, test_copy=test_copy)
        processed_random_basket_split_path = os.path.join(processed_random_basket_split_path, parameterized_path)
        if not os.path.exists(processed_random_basket_split_path):
            if test_rate == 0.1 and random is False and n_negative == 2 and by_user is False and test_copy == 10:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(url=self.processed_random_basket_split_url, path=processed_random_basket_split_path)
                folder.download()
            else:
                # make
                self.make_random_basket_split(test_rate=test_rate, random=random, n_negative=n_negative, by_user=by_user,
                                              test_copy=test_copy)

        # load data from local storage
        train_file = os.path.join(processed_random_basket_split_path, "train.npz")
        train_data = self.get_dataframe_from_npz(train_file)

        valid_file = os.path.join(processed_random_basket_split_path, "valid_0.npz")
        valid_data = self.get_dataframe_from_npz(valid_file)

        test_file = os.path.join(processed_random_basket_split_path, "test_0.npz")
        test_data = self.get_dataframe_from_npz(test_file)
        return train_data, valid_data, test_data

    def load_temporal_split(self, test_rate=0.1, n_negative=2, by_user=False, test_copy=10):
        """load split date generated by temporal_split.

        Load split data generated by temporal_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - Ture: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        processed_temporal_split_path = os.path.join(self.processed_path, "temporal")
        if not os.path.exists(processed_temporal_split_path):
            os.mkdir(processed_temporal_split_path)

        parameterized_path = generate_parameterized_path(test_rate=test_rate, random=False, n_negative=n_negative,
                                                         by_user=by_user, test_copy=test_copy)
        processed_temporal_split_path = os.path.join(processed_temporal_split_path, parameterized_path)
        if not os.path.exists(processed_temporal_split_path):
            if test_rate == 0.1 and n_negative == 2 and by_user is False and test_copy == 10:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(url=self.processed_temporal_split_url, path=processed_temporal_split_path)
                folder.download()
            else:
                # make
                self.make_temporal_split(test_rate=test_rate, n_negative=n_negative,
                                         by_user=by_user, test_copy=test_copy)

        # load data from local storage
        train_file = os.path.join(processed_temporal_split_path, "train.npz")
        train_data = self.get_dataframe_from_npz(train_file)

        valid_file = os.path.join(processed_temporal_split_path, "valid_0.npz")
        valid_data = self.get_dataframe_from_npz(valid_file)

        test_file = os.path.join(processed_temporal_split_path, "test_0.npz")
        test_data = self.get_dataframe_from_npz(test_file)
        return train_data, valid_data, test_data

    def load_temporal_basket_split(self, test_rate=0.1, n_negative=2, by_user=False, test_copy=10):
        """load split date generated by temporal_basket_split.

        Load split data generated by temporal_basket_split from Onedrive, with test_rate = 0.1 and by_user = False.

        Args:
            test_rate: percentage of the test data. Note that percentage of the validation data will be the same as testing.
            n_negative:  Number of negative samples for testing and validation data.
            by_user: bool. Default False.
                    - True: user-based split,
                    - False: global split,
            test_copy: int. Default 10. The number of testing and validation copies.

        Returns:
            Dataframe that have already by labeled by a col with "train", "test" or "valid".
        """
        processed_temporal_basket_split_path = os.path.join(self.processed_path, "temporal_basket")
        if not os.path.exists(processed_temporal_basket_split_path):
            os.mkdir(processed_temporal_basket_split_path)

        parameterized_path = generate_parameterized_path(test_rate=test_rate, random=False, n_negative=n_negative,
                                                         by_user=by_user, test_copy=test_copy)
        processed_temporal_basket_split_path = os.path.join(processed_temporal_basket_split_path, parameterized_path)
        if not os.path.exists(processed_temporal_basket_split_path):
            if test_rate == 0.1 and n_negative == 2 and by_user is False and test_copy == 10:
                # default parameters, can be downloaded from Onedrive
                folder = OneDrive(url=self.processed_temporal_basket_split_url,
                                  path=processed_temporal_basket_split_path)
                folder.download()
            else:
                # make
                self.make_temporal_basket_split(test_rate=test_rate, n_negative=n_negative,
                                                by_user=by_user, test_copy=test_copy)

        # load data from local storage
        train_file = os.path.join(processed_temporal_basket_split_path, "train.npz")
        train_data = self.get_dataframe_from_npz(train_file)

        valid_file = os.path.join(processed_temporal_basket_split_path, "valid_0.npz")
        valid_data = self.get_dataframe_from_npz(valid_file)

        test_file = os.path.join(processed_temporal_basket_split_path, "test_0.npz")
        test_data = self.get_dataframe_from_npz(test_file)
        return train_data, valid_data, test_data
