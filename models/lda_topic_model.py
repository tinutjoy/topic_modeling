"""Defines LDATopicModel class"""
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.models import ldamulticore

from configurations.lda_configuration import lda_model


class LDATopicModel:
    """
    Implements LDA model.
    """
    def __init__(self, local_path: str, num_topics: int, model_params: Optional[Dict[str, Any]] = None):
        self.local_path = local_path
        self.num_topics = num_topics
        if model_params is None:
            model_params = lda_model['model_parameters']
        self.model_params = model_params

    def fit(self, bow_corpus: List[List[Tuple[int, int]]], data_dictionary: Dictionary) -> ldamodel.LdaModel:
        """
        Fit Gensim's LDA model on training corpus.
        """
        return ldamulticore.LdaMulticore(corpus=bow_corpus, id2word=data_dictionary,
                                         num_topics=self.num_topics, **self.model_params)

    def save(self,  model, model_name: str = None):
        model_path = os.path.join(self.local_path, "model")
        os.makedirs(model_path, exist_ok=True)
        if model_name is None:
            model_name = "lda_model"
        model.save(os.path.join(model_path, model_name + ".model"))
