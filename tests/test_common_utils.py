import unittest

from beta_rec.utils.common_util import str2bool


class TestCommonUtil(unittest.TestCase):
    """TestCommonUtil Class."""

    def test_str2bool(self):
        """Test convert from string to boolean."""
        self.assertEqual(str2bool("1"), True)
        self.assertEqual(str2bool("0"), False)
        self.assertEqual(str2bool("yes"), True)
        self.assertEqual(str2bool("no"), False)
        self.assertEqual(str2bool("Yes"), True)
        self.assertEqual(str2bool("No"), False)
        self.assertEqual(str2bool("True"), True)
        self.assertEqual(str2bool("false"), False)
        self.assertEqual(str2bool("y"), True)
        self.assertEqual(str2bool("N"), False)
