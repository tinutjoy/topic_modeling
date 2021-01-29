"""Defines DataFetcher class"""
from typing import List, Tuple, Optional

from sklearn.datasets import fetch_20newsgroups


class DataFetcher:
    """
    Implements different data fetching methods.
    """
    def __init__(self, shuffle: bool = True, seed: Optional[int] = 1, remove_fields: Optional[Tuple[str]] = None):
        self.shuffle = shuffle
        self.seed = seed
        if remove_fields is not None:
            self.remove_fields = remove_fields
        else:
            self.remove_fields = ()

    def fetch_20news_data(self) -> List[str]:
        """
        Fetches 20news dataset.
        """
        try:
            fetcher = fetch_20newsgroups(shuffle=True, random_state=self.seed, remove=self.remove_fields)
            return fetcher.data
        except Exception:
            raise ValueError("Issue with downloading 20news dataset")
