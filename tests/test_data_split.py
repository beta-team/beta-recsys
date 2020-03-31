import unittest
from unittest import mock
import numpy as np
import numpy
from src.datasets.data_split import *

def generate_temporal_testdata():
    data = {
        DEFAULT_USER_COL: [0, 1, 0, 0, 0, 1, 1, 0, 2, 2],
        DEFAULT_ORDER_COL: [0, 1, 0, 0, 2, 3, 4, 0, 5, 5],
        DEFAULT_TIMESTAMP_COL: [100, 200, 100, 100, 300, 400, 410, 100, 500, 500],
        DEFAULT_ITEM_COL: [10, 10, 20, 30, 40, 50, 40, 60, 60, 10],
        DEFAULT_RATING_COL: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    data = pd.DataFrame(data)
    return data

def generate_data():
    data = {
        DEFAULT_USER_COL: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        DEFAULT_ORDER_COL: [0, 1, 2, 2, 3, 4, 5, 6, 7, 8],
        DEFAULT_TIMESTAMP_COL: [100, 200, 300, 300, 400, 500, 600, 700, 800, 900],
        DEFAULT_ITEM_COL: [10, 20, 20, 30, 40, 50, 40, 60, 60, 10],
        DEFAULT_RATING_COL: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    data = pd.DataFrame(data)
    return data

def generate_data_by_user():
    data = {
        DEFAULT_USER_COL: [0, 0, 0, 0, 1, 1, 1, 1],
        DEFAULT_ORDER_COL: [0, 1, 2, 2, 3, 4, 5, 5],
        DEFAULT_TIMESTAMP_COL: [100, 200, 300, 300, 400, 500, 600, 600],
        DEFAULT_ITEM_COL: [10, 20, 20, 30, 40, 50, 40, 60],
        DEFAULT_RATING_COL: [1, 1, 1, 1, 1, 1, 1, 1],
    }
    data = pd.DataFrame(data)
    return data

class TestDataSplit(unittest.TestCase):
        
    def test_temporal_split(self):
        testdata = generate_temporal_testdata()
        
        # check shape
        self.assertEqual(testdata.shape[0], 10)
        self.assertEqual(testdata.shape[1], 5)
        
        data = temporal_split(testdata, test_rate=0.2, by_user=False)
        # check shape
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 6)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 2)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 2)
        
        # check validate
        self.assertEqual(tp_validate.iloc[0,0], 1)
        self.assertEqual(tp_validate.iloc[0,1], 3)
        self.assertEqual(tp_validate.iloc[0,2], 400)
        self.assertEqual(tp_validate.iloc[0,3], 50)
        self.assertEqual(tp_validate.iloc[1,0], 1)
        self.assertEqual(tp_validate.iloc[1,1], 4)
        self.assertEqual(tp_validate.iloc[1,2], 410)
        self.assertEqual(tp_validate.iloc[1,3], 40)
        
        # check test
        self.assertEqual(tp_test.iloc[0,0], 2)
        self.assertEqual(tp_test.iloc[0,1], 5)
        self.assertEqual(tp_test.iloc[0,2], 500)
        self.assertEqual(tp_test.iloc[0,3], 60)
        self.assertEqual(tp_test.iloc[1,0], 2)
        self.assertEqual(tp_test.iloc[1,1], 5)
        self.assertEqual(tp_test.iloc[1,2], 500)
        self.assertEqual(tp_test.iloc[1,3], 10)
        
    def test_temporal_split_by_user(self):
        testdata = generate_data_by_user()
        
        # check shape
        self.assertEqual(testdata.shape[0], 8)
        self.assertEqual(testdata.shape[1], 5)
        
        data = temporal_split(testdata, test_rate=0.1, by_user=True)
        # check shape
        self.assertEqual(data.shape[0], 8)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 4)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 2)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 2)
        
        # check validate
        self.assertEqual(tp_validate.iloc[0,0], 0)
        self.assertEqual(tp_validate.iloc[0,1], 2)
        self.assertEqual(tp_validate.iloc[0,2], 300)
        self.assertEqual(tp_validate.iloc[0,3], 20)
        self.assertEqual(tp_validate.iloc[1,0], 1)
        self.assertEqual(tp_validate.iloc[1,1], 5)
        self.assertEqual(tp_validate.iloc[1,2], 600)
        self.assertEqual(tp_validate.iloc[1,3], 40)
        
        # check test
        self.assertEqual(tp_test.iloc[0,0], 0)
        self.assertEqual(tp_test.iloc[0,1], 2)
        self.assertEqual(tp_test.iloc[0,2], 300)
        self.assertEqual(tp_test.iloc[0,3], 30)
        self.assertEqual(tp_test.iloc[1,0], 1)
        self.assertEqual(tp_test.iloc[1,1], 5)
        self.assertEqual(tp_test.iloc[1,2], 600)
        self.assertEqual(tp_test.iloc[1,3], 60)
    
    def test_temporal_basket_split(self):
        testdata = generate_temporal_testdata()
        
        data = temporal_basket_split(testdata, test_rate=0.1, by_user=False)
        # check shape
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 7)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 1)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 2)
        
        # check validate
        self.assertEqual(tp_validate.iloc[0, 0], 1)
        self.assertEqual(tp_validate.iloc[0, 1], 4)
        self.assertEqual(tp_validate.iloc[0, 2], 410)
        self.assertEqual(tp_validate.iloc[0, 3], 40)
        
        # check test
        self.assertEqual(tp_test.iloc[0, 0], 2)
        self.assertEqual(tp_test.iloc[0, 1], 5)
        self.assertEqual(tp_test.iloc[0, 2], 500)
        self.assertEqual(tp_test.iloc[0, 3], 60)
        self.assertEqual(tp_test.iloc[1, 0], 2)
        self.assertEqual(tp_test.iloc[1, 1], 5)
        self.assertEqual(tp_test.iloc[1, 2], 500)
        self.assertEqual(tp_test.iloc[1, 3], 10)
        
    def test_temporal_basket_split_by_user(self):
        testdata = generate_data_by_user()
        
        data = temporal_basket_split(testdata, test_rate=0.1, by_user=True)
        # check shape
        self.assertEqual(data.shape[0], 8)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 2)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 2)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 4)
        
        # check validate
        self.assertEqual(tp_validate.iloc[0, 0], 0)
        self.assertEqual(tp_validate.iloc[0, 1], 1)
        self.assertEqual(tp_validate.iloc[0, 2], 200)
        self.assertEqual(tp_validate.iloc[0, 3], 20)
        self.assertEqual(tp_validate.iloc[1, 0], 1)
        self.assertEqual(tp_validate.iloc[1, 1], 4)
        self.assertEqual(tp_validate.iloc[1, 2], 500)
        self.assertEqual(tp_validate.iloc[1, 3], 50)
        
        # check test
        self.assertEqual(tp_test.iloc[0, 0], 0)
        self.assertEqual(tp_test.iloc[0, 1], 2)
        self.assertEqual(tp_test.iloc[0, 2], 300)
        self.assertEqual(tp_test.iloc[0, 3], 20)
        self.assertEqual(tp_test.iloc[1, 0], 0)
        self.assertEqual(tp_test.iloc[1, 1], 2)
        self.assertEqual(tp_test.iloc[1, 2], 300)
        self.assertEqual(tp_test.iloc[1, 3], 30)
        self.assertEqual(tp_test.iloc[2, 0], 1)
        self.assertEqual(tp_test.iloc[2, 1], 5)
        self.assertEqual(tp_test.iloc[2, 2], 600)
        self.assertEqual(tp_test.iloc[2, 3], 40)
        self.assertEqual(tp_test.iloc[3, 0], 1)
        self.assertEqual(tp_test.iloc[3, 1], 5)
        self.assertEqual(tp_test.iloc[3, 2], 600)
        self.assertEqual(tp_test.iloc[3, 3], 60)
        
    def test_leave_one_item(self):
        testdata = generate_data()
        
        data = leave_one_out(testdata)
        # check shape
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 4)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 3)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 3)
        
         # check validate
        self.assertEqual(tp_validate.iloc[0, 0], 0)
        self.assertEqual(tp_validate.iloc[0, 1], 2)
        self.assertEqual(tp_validate.iloc[0, 2], 300)
        self.assertEqual(tp_validate.iloc[0, 3], 20)
        self.assertEqual(tp_validate.iloc[1, 0], 1)
        self.assertEqual(tp_validate.iloc[1, 1], 4)
        self.assertEqual(tp_validate.iloc[1, 2], 500)
        self.assertEqual(tp_validate.iloc[1, 3], 50)
        self.assertEqual(tp_validate.iloc[2, 0], 2)
        self.assertEqual(tp_validate.iloc[2, 1], 7)
        self.assertEqual(tp_validate.iloc[2, 2], 800)
        self.assertEqual(tp_validate.iloc[2, 3], 60)
        
        # check test
        self.assertEqual(tp_test.iloc[0, 0], 0)
        self.assertEqual(tp_test.iloc[0, 1], 2)
        self.assertEqual(tp_test.iloc[0, 2], 300)
        self.assertEqual(tp_test.iloc[0, 3], 30)
        self.assertEqual(tp_test.iloc[1, 0], 1)
        self.assertEqual(tp_test.iloc[1, 1], 5)
        self.assertEqual(tp_test.iloc[1, 2], 600)
        self.assertEqual(tp_test.iloc[1, 3], 40)
        self.assertEqual(tp_test.iloc[2, 0], 2)
        self.assertEqual(tp_test.iloc[2, 1], 8)
        self.assertEqual(tp_test.iloc[2, 2], 900)
        self.assertEqual(tp_test.iloc[2, 3], 10)
        
    def test_leave_one_basket(self):
        testdata = generate_data()
        
        data = leave_one_basket(testdata)
        # check shape
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 3)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 3)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 4)
        
        # check validate
        self.assertEqual(tp_validate.iloc[0, 0], 0)
        self.assertEqual(tp_validate.iloc[0, 1], 1)
        self.assertEqual(tp_validate.iloc[0, 2], 200)
        self.assertEqual(tp_validate.iloc[0, 3], 20)
        self.assertEqual(tp_validate.iloc[1, 0], 1)
        self.assertEqual(tp_validate.iloc[1, 1], 4)
        self.assertEqual(tp_validate.iloc[1, 2], 500)
        self.assertEqual(tp_validate.iloc[1, 3], 50)
        self.assertEqual(tp_validate.iloc[2, 0], 2)
        self.assertEqual(tp_validate.iloc[2, 1], 7)
        self.assertEqual(tp_validate.iloc[2, 2], 800)
        self.assertEqual(tp_validate.iloc[2, 3], 60)
        
        # check test
        self.assertEqual(tp_test.iloc[0, 0], 0)
        self.assertEqual(tp_test.iloc[0, 1], 2)
        self.assertEqual(tp_test.iloc[0, 2], 300)
        self.assertEqual(tp_test.iloc[0, 3], 20)
        self.assertEqual(tp_test.iloc[1, 0], 0)
        self.assertEqual(tp_test.iloc[1, 1], 2)
        self.assertEqual(tp_test.iloc[1, 2], 300)
        self.assertEqual(tp_test.iloc[1, 3], 30)
        self.assertEqual(tp_test.iloc[2, 0], 1)
        self.assertEqual(tp_test.iloc[2, 1], 5)
        self.assertEqual(tp_test.iloc[2, 2], 600)
        self.assertEqual(tp_test.iloc[2, 3], 40)
        self.assertEqual(tp_test.iloc[3, 0], 2)
        self.assertEqual(tp_test.iloc[3, 1], 8)
        self.assertEqual(tp_test.iloc[3, 2], 900)
        self.assertEqual(tp_test.iloc[3, 3], 10)
        
    def my_shuffle_side_effect(*args, **kwargs):
        a = args[1]
        first, last = a[0], a[len(a) - 1]
        temp = a[0].copy()
        a[0] = last
        a[len(a) - 1] = temp
        return a
    
    @mock.patch('sklearn.utils.shuffle')
    def test_random_split(self, mock_shuffle):
        testdata = generate_data()
        
        # mock sklearn.utils.shuffle
        mock_shuffle.side_effect = self.my_shuffle_side_effect
        
        data = random_split(testdata, test_rate=0.1, by_user=False)
        # check shape
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 8)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 1)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 1)
        
        # check validate
        self.assertEqual(tp_validate.iloc[0, 0], 2)
        self.assertEqual(tp_validate.iloc[0, 1], 7)
        self.assertEqual(tp_validate.iloc[0, 2], 800)
        self.assertEqual(tp_validate.iloc[0, 3], 60)
        
        # check test
        self.assertEqual(tp_test.iloc[0, 0], 0)
        self.assertEqual(tp_test.iloc[0, 1], 0)
        self.assertEqual(tp_test.iloc[0, 2], 100)
        self.assertEqual(tp_test.iloc[0, 3], 10)
        
    @mock.patch('sklearn.utils.shuffle')    
    def test_random_basket_split(self, mock_shuffle):
        testdata = generate_data()
        
        # mock sklearn.utils.shuffle
        mock_shuffle.side_effect = self.my_shuffle_side_effect
        
        data = random_basket_split(testdata, test_rate=0.1, by_user=False)
        # check shape
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 6)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 8)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 1)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 1)
        
        # check number
        tp_train = data[data[DEFAULT_FLAG_COL] == "train"]
        self.assertEqual(tp_train.shape[0], 8)
        tp_validate = data[data[DEFAULT_FLAG_COL] == "validate"]
        self.assertEqual(tp_validate.shape[0], 1)
        tp_test = data[data[DEFAULT_FLAG_COL] == "test"]
        self.assertEqual(tp_test.shape[0], 1)
        
        # check validate
        self.assertEqual(tp_validate.iloc[0, 0], 2)
        self.assertEqual(tp_validate.iloc[0, 1], 7)
        self.assertEqual(tp_validate.iloc[0, 2], 800)
        self.assertEqual(tp_validate.iloc[0, 3], 60)
        
        # check test
        self.assertEqual(tp_test.iloc[0, 0], 0)
        self.assertEqual(tp_test.iloc[0, 1], 0)
        self.assertEqual(tp_test.iloc[0, 2], 100)
        self.assertEqual(tp_test.iloc[0, 3], 10)