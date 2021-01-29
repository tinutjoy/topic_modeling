"""Defines DataPreprocessor class"""
from typing import List
from typing import Optional
from typing import Sequence

from string import punctuation

from nltk.tokenize.api import TokenizerI
from nltk import RegexpTokenizer
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
from nltk import corpus as nltk_corpus

STOP_WORDS_EN = set(nltk_corpus.stopwords.words('english'))


class DataPreprocessor:
    """
    This class preprocesses the raw textual data and prepares them to be consumed by Gensim.
    """

    def __init__(self,
                 stemmer: Optional[StemmerI] = None,
                 tokenizer: Optional[TokenizerI] = None,
                 custom_stop_words: Optional[Sequence[str]] = None):
        if stemmer is None:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = stemmer
        if custom_stop_words is None:
            self.stop_words = STOP_WORDS_EN
        else:
            self.stop_words = custom_stop_words
        self.tokenizer = tokenizer

    def tokenize(self, text, pattern=r'\s+', gaps=True):
        """
        Tokenize the sentences.
        """
        if self.tokenizer is None:
            self.tokenizer = RegexpTokenizer(pattern=pattern, gaps=gaps)
        return [token.strip() for token in self.tokenizer.tokenize(text)]

    def remove_stopwords(self, tokens):
        """
        removes stop words.
        """
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens):
        """
        lemmatize the tokens.
        """
        return [self.stemmer.stem(token) for token in tokens]

    @staticmethod
    def clean_text(text):
        """
        cleans the raw text by removing punctuations.
        """
        remove_punc = {ord(p): u" " for p in punctuation}
        return text.lower().translate(remove_punc)

    @staticmethod
    def remove_short_tokens(tokens):
        """
        removes short tokens.
        """
        return [token for token in tokens if len(token) > 2]

    def process_text(self, documents: List[str]) -> List[List[str]]:
        """
        Processes a list of documents.
        :param documents:
        :return: lemma_list: List of lemmas.
        """
        cleaned_docs = []
        for data in documents:
            clean_txt = self.clean_text(data)
            tokens = self.tokenize(clean_txt)
            valid_tokens = self.remove_stopwords(tokens)
            stemmed_tokens = self.stem_tokens(valid_tokens)
            final_tokens = self.remove_short_tokens(stemmed_tokens)
            cleaned_docs.append(final_tokens)
        return cleaned_docs
