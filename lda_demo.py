"""demo topic modeling using LDA"""
import json
import tempfile
from nltk import download

from typing import List
from pathlib import Path

from data.data_fetcher import DataFetcher
from preprocess.data_preprocessor import DataPreprocessor
from features.feature_extractor import FeatureExtractor
from models.lda_topic_model import LDATopicModel

# TODO(@tinu) Move to set up.
download('stopwords', quiet=True)


def demo_lda(data: List[str]) -> None:
    # TODO(@tinu) Handle this better, create the results folder somewhere else.
    results_folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.as_posix())
    preprocessed_data = DataPreprocessor().process_text(data)
    corpus_bow, corpus_dictionary = FeatureExtractor(results_folder.name).generate_bow_features(preprocessed_data)
    model = LDATopicModel(num_topics=20, local_path=results_folder.name).fit(corpus_bow, corpus_dictionary)
    # TODO(@tinu) Provide a better way to consume/represent the output.
    print(json.dumps(model.print_topics(num_topics=20)))

    # predict the topic for a new text
    text_data = ["Coronavirus Australia news: Greg Hunt praises Australia's 'extraordinary efforts' "
                 "after 12 days of no community transmission — as it happened"]
    text_preprocessed = DataPreprocessor().process_text(text_data)
    bow_corpus = corpus_dictionary.doc2bow(text_preprocessed)
    predictions = model.get_document_topics(bow_corpus)
    # TODO(@tinu) Provide a better way to represent the output.
    print(predictions)
    results_folder.cleanup()


if __name__ == '__main__':
    dataset = DataFetcher().fetch_20news_data()
    demo_lda(dataset)
