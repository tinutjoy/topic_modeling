"""Defines FeatureExtractor class"""

import os
from typing import List

from gensim import corpora
from gensim.models import Phrases
from gensim.models import phrases


class FeatureExtractor:
    """
    Extracts bag of words features from corpus for LDA model.
    """
    def __init__(self, local_path: str):
        self.local_path = local_path

    def generate_bow_features(self,
                              documents: List[List[str]],
                              filter_vocab: bool = True,
                              no_below: int = 5,
                              no_above: float = 0.3,
                              bigram_min_count: int = 5,
                              bigram_threshold: int = 5) -> [corpora.MmCorpus, corpora.Dictionary]:
        """
        Generates bag_of_words representation from the training documents.
        """
        bigram = Phrases(documents, min_count=bigram_min_count, threshold=bigram_threshold)
        bigram_mod = phrases.Phraser(bigram)
        data_words_bigrams = [bigram_mod[doc] for doc in documents]
        docs_dict = corpora.Dictionary(data_words_bigrams)
        if filter_vocab:
            docs_dict.filter_extremes(no_below=no_below, no_above=no_above)
            docs_dict.compactify()
        docs_corpus = [docs_dict.doc2bow(doc) for doc in data_words_bigrams]
        return docs_corpus, docs_dict

    def save_dictionary_corpus(self, dictionary, corpus) -> None:
        os.makedirs(self.local_path, exist_ok=True)
        dict_path = os.path.join(self.local_path, "dictionary.gensim")
        corpus_path = os.path.join(self.local_path,  "corpus.mm")
        dictionary.save(dict_path)
        corpora.MmCorpus.serialize(corpus_path, corpus)
