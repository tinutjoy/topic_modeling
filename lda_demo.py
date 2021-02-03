"""demo topic modeling using LDA"""
import os
import json
import argparse

from typing import List

from data.data_fetcher import DataFetcher
from preprocess.data_preprocessor import DataPreprocessor
from features.feature_extractor import FeatureExtractor
from models.lda_topic_model import LDATopicModel


def demo_lda(data: List[str], num_topics: int, local_folder: str, text: str) -> None:
    os.makedirs(local_folder, exist_ok=True)
    preprocessed_data = DataPreprocessor().process_text(data)
    corpus_bow, corpus_dictionary = FeatureExtractor(local_folder).generate_bow_features(preprocessed_data)
    model = LDATopicModel(num_topics=num_topics, local_path=local_folder).fit(corpus_bow, corpus_dictionary)
    # TODO(@tinu) Provide a better way to represent the output.
    # Printing number of topics and the words constituting the topics extracted from the corpus.
    representative_topic_words = model.print_topics(num_topics=num_topics, num_words=5)
    print(json.dumps(representative_topic_words, indent=4))

    # predict the topic for a new text
    text_preprocessed = DataPreprocessor().process_text([text])
    bow_corpus = corpus_dictionary.doc2bow(text_preprocessed[0])
    # It predicts (with a probability) the topic ids that the text belongs to.
    predictions = model.get_document_topics(bow_corpus)
    # TODO(@tinu) Provide a better way to represent the output.
    print(predictions)


if __name__ == '__main__':
    dataset = DataFetcher().fetch_20news_data()
    parser = argparse.ArgumentParser(description='Demo of topic model training and prediction')
    parser.add_argument('-d',
                        '--experiment_directory',
                        type=str,
                        required=True,
                        help="To store the model, corpus and dictionary")
    parser.add_argument('-n',
                        '--num_topics',
                        type=int,
                        required=False,
                        default=10,
                        help="Number of topics to be extracted from data")
    parser.add_argument('-t',
                        '--text',
                        type=str,
                        required=False,
                        default="Toyota produces different models of cars.",
                        help="text for predicting the topics")
    args = parser.parse_args()

    demo_lda(dataset, args.num_topics, args.experiment_directory, args.text)
