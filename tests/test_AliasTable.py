import unittest
from unittest import mock

from beta_rec.utils.alias_table import AliasTable


class TestAliasTable(unittest.TestCase):
    """TestAliasTable Class."""

    @mock.patch("numpy.random.randint")
    @mock.patch("numpy.random.uniform")
    def test_list(self, mock_uniform, mock_randint):
        """Test AliasTable with list as input."""
        obj_freq = [6, 4, 1, 1]
        self.aliasTable = AliasTable(obj_freq)

        # check vocab_size
        self.assertEqual(self.aliasTable.vocab_size, 4)

        # check prob_arr
        self.assertEqual(round(self.aliasTable.prob_arr[0], 2), 1.0)
        self.assertEqual(round(self.aliasTable.prob_arr[1], 2), 0.67)
        self.assertEqual(round(self.aliasTable.prob_arr[2], 2), 0.33)
        self.assertEqual(round(self.aliasTable.prob_arr[3], 2), 0.33)

        # check alias_arr
        self.assertEqual(self.aliasTable.alias_arr[0], 0)
        self.assertEqual(self.aliasTable.alias_arr[1], 0)
        self.assertEqual(self.aliasTable.alias_arr[2], 0)
        self.assertEqual(self.aliasTable.alias_arr[3], 1)

        mock_uniform.side_effect = [0.1, 0.64, 0.8, 0.6]
        mock_randint.return_value = [2, 2, 0, 1]

        result = self.aliasTable.sample(4)
        self.assertEqual(result[0], 2)
        self.assertEqual(result[1], 0)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)

    @mock.patch("numpy.random.randint")
    @mock.patch("numpy.random.uniform")
    def test_dict(self, mock_uniform, mock_randint):
        """Test AliasTable with dictionary as input."""
        obj_freq = {100: 6, 102: 4, 103: 1, 104: 1}
        self.aliasTable = AliasTable(obj_freq)

        # check vacab_size
        self.assertEqual(self.aliasTable.vocab_size, 4)

        # check prob_arr
        self.assertEqual(round(self.aliasTable.prob_arr[0], 2), 1.0)
        self.assertEqual(round(self.aliasTable.prob_arr[1], 2), 0.67)
        self.assertEqual(round(self.aliasTable.prob_arr[2], 2), 0.33)
        self.assertEqual(round(self.aliasTable.prob_arr[3], 2), 0.33)

        # check alias_arr
        # check alias_arr
        self.assertEqual(self.aliasTable.alias_arr[0], 0)
        self.assertEqual(self.aliasTable.alias_arr[1], 0)
        self.assertEqual(self.aliasTable.alias_arr[2], 0)
        self.assertEqual(self.aliasTable.alias_arr[3], 1)

        mock_uniform.side_effect = [0.1, 0.64, 0.8, 0.6]
        mock_randint.return_value = [2, 2, 0, 1]

        result = self.aliasTable.sample(4)
        self.assertEqual(result[0], 103)
        self.assertEqual(result[1], 100)
        self.assertEqual(result[2], 100)
        self.assertEqual(result[3], 102)

    def test_2dim_list(self):
        """Test AliasTable with 2-dimension list as input."""
        obj_freq = [[1, 2, 3], [4, 5, 6]]
        try:
            self.aliasTable = AliasTable(obj_freq)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
