import unittest

from beta_rec.utils.download import get_format


class TestDownload(unittest.TestCase):
    """TestDownload Class."""

    def test_get_format(self):
        """Test get file format from filename postfix."""
        self.assertEqual(get_format("gz"), "gztar")
        self.assertEqual(get_format("bz2"), "bztar")
