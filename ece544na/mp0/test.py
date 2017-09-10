"""Unit Tests examples for mp0.

python run_tests.py
"""

import unittest

from io_tools import read_data_from_file
from data_tools import title_cleanup


class IoToolsTest(unittest.TestCase):
    def test_first_element(self):
        data = read_data_from_file("data/news_data.txt")
        self.assertEqual("Yields on CDs Fell in the Latest Week", data[000][0])
        self.assertEqual(3.0, data[000][1])


class DataToolsTest(unittest.TestCase):
    def setUp(self):
        self.data = {}
        self.data[000] = ["ECE544 is grEAt!?", 9.9]

    def test_title_clean_up(self):
        title_cleanup(self.data)
        self.assertEqual("ece is great", self.data[000][0])


if __name__ == '__main__':
    unittest.main()
