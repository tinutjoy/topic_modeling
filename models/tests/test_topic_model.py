"""Tests for LDATopicModel"""
import unittest
import tempfile
import numpy as np
from pathlib import Path

from gensim.models.ldamodel import LdaModel

from models.lda_topic_model import LDATopicModel


bow = [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 1), (4, 1), (5, 1), (6, 1)]]
id2word = {0: 'Tom', 1: 'a', 2: 'duck', 3: 'saw', 4: 'He', 5: 'shot', 6: 'the'}
num_topics = 2


class TestLDATopicModel(unittest.TestCase):
    def setUp(self):
        self.test_folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.as_posix())
        self.topic_model = LDATopicModel(self.test_folder.name, num_topics)

    def test_fit_save_model(self):
        """
        Test model fitting and serialization.
        """
        model = self.topic_model.fit(bow, id2word)
        self.assertEqual(model.num_topics, num_topics)
        assert model.num_topics is num_topics
        self.assertDictEqual(dict(model.id2word), id2word)

        self.topic_model.save(model)
        loaded_model = LdaModel.load(Path(self.test_folder.name).joinpath('model', 'lda_model.model').as_posix())
        self.assertEqual(type(loaded_model), type(model))
        self.assertEqual(loaded_model.id2word, model.id2word)
        self.assertEqual(loaded_model.num_topics, model.num_topics)
        np.allclose(loaded_model.expElogbeta, model.expElogbeta)

    def tearDown(self) -> None:
        self.test_folder.cleanup()
