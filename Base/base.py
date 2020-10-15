import gensim.models as models
from gensim.models import Word2Vec, FastText
import multiprocessing as mp


class BaseEmbedding(object):
    def __init__(self, model_type, n_dimensions=300, window_size=3, ):
        self.model_type = self._check_model(model_type)
        self.model = None

        # embedding model parameters
        self.n_dimensions = n_dimensions
        self.window_size = window_size


    def _check_model(self, model_type):
        return model_type

    def fit(self, sentences):
        if self.model_type == 'word2vec_CBOW':
            self.model = Word2Vec(sentences=sentences, size=300, window=3,
                                  min_count=2, sg=0,
                                  workers=mp.cpu_count(), sample=0.001)
        elif self.model_type == 'word2vec_skipgram':
            self.model = Word2Vec(sentences=sentences, size=300, window=3,
                                  min_count=2, sg=1,
                                  workers=mp.cpu_count(), sample=0.001)

        elif self.model_type == 'fasttext':
            self.model = Word2Vec(sentences=sentences, size=300, window=3,
                                  min_count=2, workers=mp.cpu_count())

    def save(self, path):
        self.model.wv.save(path)
