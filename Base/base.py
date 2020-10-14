import gensim.models as models
from gensim.models import Word2Vec, FastText
import multiprocessing as mp


class BaseEmbedding(object):
    def __init__(self, model_type, ):
        self.model_type = model_type
        self.model = None

    def _check_model(self):
        pass

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
