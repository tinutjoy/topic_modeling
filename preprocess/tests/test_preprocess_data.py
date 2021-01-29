"""Tests for DataPreprocessor"""
import unittest
from nltk import download

# TODO(@tinu) Make sure not to download the stopwords everytime the script is run. May be push it to Pipfile.
download('stopwords', quiet=True)

from preprocess.data_preprocessor import DataPreprocessor


test_stop_words = [["it", "is", "interesting"],["a", "job", "good"]]
expected_test_stop_words = [["interesting"], ["job","good"]]

test_short_tokens = [["it", "is", "interesting"],["a", "job", "good"]]
expected_short_tokens = [["interesting"], ["job", "good"]]

test_tokens_stemming = [["worked", "at", "things"], ["playing", "soccer", "runs"]]
expected_stemmed_tokens = [["work", "at", "thing"], ["play", "soccer", "run"]]


test_tokens_text = ["This is interesting", "One is a number"]
expected_tokens = [["This", "is", "interesting"], ["One", "is", "a", "number"]]

test_stemming = ["This is interesting", "One is a number"]
expected_test_stemming = ["interesting, number"]


test_raw_text = ["@.#&$This is interesting", "One @.#&$ is @.#&$ a @.#&$ number"]
expected_clean_text = ["this is interesting", "one is a number"]
expected_processed_text = [["interest"], ["one", "number"]]


class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()

    def test_remove_stopwords(self):
        """
        Test removing stopwords.
        """
        for test_stop_word, expected_text in zip(test_stop_words, expected_test_stop_words):
            with self.subTest(test_stop_word=test_stop_word, expected_text=expected_text):
                stopwords_removed_text = self.preprocessor.remove_stopwords(test_stop_word)
                self.assertEqual(stopwords_removed_text, expected_text)

    def test_remove_short_tokens(self):
        """
        Test removing short tokens.
        """
        for test_short_token, expected_short_token in zip(test_short_tokens, expected_short_tokens):
            with self.subTest(test_text=test_short_token, expected_text=expected_short_token):
                processed_token = self.preprocessor.remove_short_tokens(test_short_token)
                self.assertEqual(processed_token, expected_short_token)

    def test_clean_text(self):
        """
        Test cleaning raw text.
        """
        for test_text, expected_text in zip(test_raw_text, expected_clean_text):
            with self.subTest(test_text=test_text, expected_text=expected_text):
                clean_text = self.preprocessor.clean_text(test_text)
                self.assertEqual(clean_text.replace(" ", ""), expected_text.replace(" ", ""))

    def test_stemmer(self):
        """
        Test stemming tokens.
        """
        for test_token, expected_stemmed_token in zip(test_tokens_stemming, expected_stemmed_tokens):
            with self.subTest(test_token=test_token, expected_stemmed_token=expected_stemmed_token):
                clean_text = self.preprocessor.stem_tokens(test_token)
                self.assertEqual(clean_text, expected_stemmed_token)

    def test_tokenize(self):
        """
        Test tokenizer.
        """
        for test_text, expected_token in zip(test_tokens_text, expected_tokens):
            with self.subTest(test_text=test_text, expected_token=expected_token):
                token = self.preprocessor.tokenize(test_text)
                self.assertEqual(token, expected_token)

    def test_process_text(self):
        """
        Test processing of text.
        """
        processed = self.preprocessor.process_text(test_raw_text)
        self.assertEqual(processed, expected_processed_text)
