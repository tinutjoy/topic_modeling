"""Tests for DataFetcher"""
import unittest

from data.data_fetcher import DataFetcher
from sklearn.datasets import fetch_20newsgroups


class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = DataFetcher()

    def test_fetch_20news_data(self):
        data = self.fetcher.fetch_20news_data()
        self.assertEqual(fetch_20newsgroups(random_state=1, shuffle=True).data, data)
