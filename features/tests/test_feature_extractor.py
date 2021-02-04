"""Tests for FeatureExtractor"""
import unittest
import tempfile
from pathlib import Path

from typing import Dict, List, Any

from gensim import corpora

from features.feature_extractor import FeatureExtractor


test_documents = [["Tom", "saw", "a", "duck"], ["He", "shot", "the", "duck"]]
dict_expected_filtered: Dict[Any, Any] = {}
bow_expected_filtered: List[List[Any]] = [[], []]

bow_expected_no_filter = [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 1), (4, 1), (5, 1), (6, 1)]]
dict_expected_no_filter = {0: 'Tom', 1: 'a', 2: 'duck', 3: 'saw', 4: 'He', 5: 'shot', 6: 'the'}


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.test_folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.as_posix())
        self.extractor = FeatureExtractor(self.test_folder.name)

    def test_generate_bow_features(self):
        """
        Test bag of words features.
        """
        bow_actual, dict_actual = self.extractor.generate_bow_features(test_documents)
        self.assertEqual(bow_actual, bow_expected_filtered)
        self.assertEqual(dict(dict_actual), dict_expected_filtered)
        bow_actual, dict_actual = self.extractor.generate_bow_features(test_documents, filter_vocab=False)
        self.assertEqual(bow_actual, bow_expected_no_filter)
        self.assertEqual(dict(dict_actual), dict_expected_no_filter)

    def test_save_dictionary_corpus(self):
        """
        Test serialization of dictionary and corpus.
        """
        bow_actual, dict_actual = self.extractor.generate_bow_features(test_documents, filter_vocab=False)
        self.extractor.save_dictionary_corpus(dict_actual, bow_actual)
        loaded_dict = corpora.Dictionary.load(Path(self.test_folder.name).joinpath('dictionary.gensim').as_posix())
        loaded_corp = corpora.MmCorpus(Path(self.test_folder.name).joinpath('corpus.mm').as_posix())
        self.assertEqual(loaded_dict, dict_actual)
        self.assertEqual(list(loaded_corp), bow_actual)

    def tearDown(self) -> None:
        self.test_folder.cleanup()
