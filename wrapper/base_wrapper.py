import os
import numpy as np

from Base.base import BaseEmbedding
from preprocessing.tokenizer import tokenize_dataset, tokenize_sentence
from preprocessing.utils import get_wrapper_arguments, check_nltk_library, add_nltk_path


class BaseWrapper(object):

    def __init__(self, training_algorithm='word2vec_CBOW',
                 n_dimensions=300, window_size=3,
                 with_tokenization=True, ignore_columns=None,
                 ):
        # embedding values
        self.mat = np.array([])
        self.keys = []

        # emebedding model parameters
        self.n_dimensions = n_dimensions
        self.window_size = window_size
        self.training_algorithm = training_algorithm

        # preprocessing data values
        self.with_tokenization = with_tokenization
        self.ignore_columns = ignore_columns if ignore_columns else []

    def fit(self, df, name='test_name'):
        pass

    def fit_db(self, input_dir):
        pass

    def load_emebedding(self, embedding_file):
        pass

    def preprocess_sentence(self, sentence):
        pass

    def get_token_embedding(self, token):
        pass

    def get_sentence_embedding(self, sentence, keepNone=False):
        pass

    def get_k_nearest_token(self, sentence, k=5, distance='cosine', pref='idx', withMean=True):
        pass
