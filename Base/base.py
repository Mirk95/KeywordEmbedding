from gensim.models import Word2Vec, FastText
import multiprocessing as mp


def _check_model(model_type):
    return model_type


class BaseEmbedding(object):
    def __init__(self, model_type, n_dimensions=300, window_size=3, ):
        self.model_type = _check_model(model_type)
        self.model = None

        # embedding model parameters
        self.n_dimensions = n_dimensions
        self.window_size = window_size

    def fit(self, sentences):
        if self.model_type == 'word2vec_CBOW':
            self.model = Word2Vec(corpus_file=sentences, size=300, window=3,
                                  min_count=1, sg=0,
                                  workers=mp.cpu_count(), sample=0.001)
        elif self.model_type == 'word2vec_skipgram':
            self.model = Word2Vec(corpus_file=sentences, size=300, window=3,
                                  min_count=1, sg=1,
                                  workers=mp.cpu_count(), sample=0.001)

        elif self.model_type == 'fasttext':
            self.model = FastText(sentences=sentences, size=300, window=3,
                                  min_count=1, workers=mp.cpu_count())

    def save(self, path):
        if self.model_type == 'fasttext':
            self.model.wv.save(path)
        else:
            self.model.wv.save_word2vec_format(path, binary=False)
