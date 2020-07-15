import os
import sys
import unittest

from beta_rec.datasets.movielens import Movielens_100k

sys.path.append("../")


class TestMovielens(unittest.TestCase):
    """TestMovielens Class."""

    def test_preprocess(self):
        """Test preprocess raw data."""
        ml_100k = Movielens_100k()
        ml_100k.preprocess()
        self.assertTrue(os.path.exists(os.getcwd() + "/datasets/ml_100k/raw/ml_100k"))
        self.assertTrue(os.path.exists(os.getcwd() + "/datasets/ml_100k/raw/ml_100k"))
        self.assertTrue(
            os.path.exists(
                os.getcwd() + "/datasets/ml_100k/processed/ml_100k_interaction.npz"
            )
        )

        interactions = ml_100k.load_interaction()
        self.assertEqual(100000, interactions.shape[0])
        self.assertEqual(4, interactions.shape[1])
