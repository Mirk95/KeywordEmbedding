import os
import datetime
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from Base.base import BaseEmbedding
from preprocessing.tokenizer import tokenize_dataset, tokenize_sentence
from preprocessing.utils import get_wrapper_arguments, check_nltk_library, add_nltk_path, prepare_emb_matrix

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
OUTPUT_FORMAT = '# {:.<60} {}'


def sentence_permutation(sentences, permutation_rate):
    new_sentences = []
    # new_length = permutation_rate * 100 * len(sentences)
    permutation_rate = np.floor(permutation_rate)

    for sentence in sentences:
        new_sentences.append(sentence)

        sentence = sentence.split(' ')

        for _ in range(permutation_rate):
            np.shuffle(sentence)
            new_sentences.append(' '.join(sentence))

    return new_sentences


class BaseWrapper(object):

    def __init__(self, training_algorithm='word2vec_CBOW',
                 n_dimensions=300, window_size=3,
                 with_tokenization=True, ignore_columns=None,
                 insert_col=True, permutation_rate=10
                 ):
        # embedding values
        self.mat = np.array([])
        self.keys = []

        # embedding model parameters
        self.n_dimensions = n_dimensions
        self.window_size = window_size
        self.training_algorithm = training_algorithm

        # preprocessing data values
        self.with_tokenization = with_tokenization
        self.ignore_columns = ignore_columns if ignore_columns else []
        self.insert_col = insert_col
        self.permutation_rate = permutation_rate

    def fit(self, df, name='test_name'):

        os.makedirs('pipeline/embeddings', exist_ok=True)

        print('#' * 80)
        t_start = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Starting run.', t_start))

        # create a copy of the input dataset
        df = df.copy()

        # ignore selected columns
        if self.ignore_columns is not None:
            for col in self.ignore_columns:
                if col in df:
                    df.drop(col, axis=1, inplace=True)

        # tokenize input dataset
        if self.with_tokenization:
            print('Generating basic tokenization.')
            df = tokenize_dataset(df, stem=True)

        print('Insert semantic content inside each tuple.')

        # insert <COL> information
        if self.insert_col:
            cols = df.columns
            new_cols = []

            for col in cols:
                new_col = '__idx_{}'.format(col)
                df[new_col] = '<{}>'.format(col.upper())
                new_cols.append(new_col)
                new_cols.append(col)

            df.reindex(new_cols, axis=1)

        # insert idx__ID information
        df['__idx'] = ['idx__' + str(i) for i in df.index]

        # create sentence list
        print('Generating sentences from dataset.')
        df = df.apply(lambda x: ' '.join(x.dropna().astype(str).to_list()), axis=1)
        df = df.values

        # check permutation
        print('Using data augmentation.')
        if self.permutation_rate >= 1:
            sentence_permutation(df, self.permutation_rate)

        print('Start embedding {}.'.format(self.training_algorithm))
        base = BaseEmbedding(self.training_algorithm, n_dimensions=self.n_dimensions, window_size=self.window_size)
        base.fit(df)

        name = 'base_{}_{}.emb'.format(self.training_algorithm, name)
        name = os.path.join('pipeline/embeddings', name)
        base.save(name)

        print('End embedding {}.'.format(self.training_algorithm))

        self.mat, self.keys = prepare_emb_matrix(name)

        t_end = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Ending run.', t_end))
        dt = t_end - t_start
        print('# Time required: {}'.format(dt.total_seconds()))

        print('')
        print('File created:')
        print('1. {}'.format(name))

    def fit_db(self, input_dir):
        pass

    def load_emebedding(self, embedding_file):
        self.mat, self.keys = prepare_emb_matrix(embedding_file)

    def preprocess_sentence(self, sentence):
        if self.with_tokenization:
            sentence = tokenize_sentence(sentence, stem=True)

        return sentence

    def get_token_embedding(self, token):
        vec = None
        if token in self.keys:
            vec = self.mat[self.keys.index(token)]

        return vec

    def get_sentence_embedding(self, sentence, keep_none=False):
        vecs = []
        sentence = sentence.split(' ')
        for word in sentence:
            vec = self.get_token_embedding(word)

            if vec is not None or keep_none:
                vecs.append(vec)

        return vecs

    def get_k_nearest_token(self, sentence, k=5, distance='cosine', pref='idx', withMean=True):
        cond = [True if x.startswith(pref) else False for x in self.keys]

        emb_sentence = self.get_sentence_embedding(sentence)
        if not emb_sentence:
            return []
        emb_sentence = np.array(emb_sentence)

        if withMean:
            emb_sentence = np.mean(emb_sentence, axis=0, keepdims=True)

        if distance == 'cosine':
            distance_matrix = cosine_distances(emb_sentence, self.mat[cond])
        elif distance == 'euclidean':
            distance_matrix = euclidean_distances(emb_sentence, self.mat[cond])
        else:
            raise ValueError('Selected the wrong distance {}'.format(distance))

        if not withMean:
            distance_matrix = np.sum(distance_matrix, axis=0, keepdims=True)

        distance_matrix = distance_matrix.ravel()

        indexes = distance_matrix.argsort()[:k]
        keys = np.array(self.keys)
        new_keys = keys[cond][indexes]

        return new_keys
