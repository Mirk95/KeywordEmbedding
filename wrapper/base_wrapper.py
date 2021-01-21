import os
import random
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from Base.base import BaseEmbedding
from preprocessing.tokenizer import tokenize_dataset, tokenize_sentence
from preprocessing.utils import get_wrapper_arguments, check_nltk_library, add_nltk_path, prepare_emb_matrix

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
OUTPUT_FORMAT = '# {:.<60} {}'


def sentence_permutation(sentences, permutation_rate):
    new_sentences = []
    permutation_rate = np.floor(permutation_rate)

    for sentence in sentences:
        new_sentences.append(sentence)
        sentence = sentence.split(' ')

        for _ in range(int(permutation_rate)):
            random.shuffle(sentence)
            new_sentences.append(' '.join(sentence))

    return np.array(new_sentences)


class BaseWrapper(object):
    def __init__(self, training_algorithm='word2vec_CBOW',
                 n_dimensions=300, window_size=3,
                 with_tokenization=True, ignore_columns=None,
                 insert_col=True, permutation_rate=10
                 ):
        # Embedding values
        self.mat = np.array([])
        self.keys = []

        # Embedding model parameters
        self.n_dimensions = n_dimensions
        self.window_size = window_size
        self.training_algorithm = training_algorithm

        # Preprocessing data values
        self.with_tokenization = with_tokenization
        self.ignore_columns = ignore_columns if ignore_columns else []
        self.insert_col = insert_col
        self.permutation_rate = permutation_rate

    def generate_walks(self, dataframe, dataset_name):
        # Ignore selected columns
        if self.ignore_columns is not None:
            for col in self.ignore_columns:
                if col in dataframe:
                    dataframe.drop(col, axis=1, inplace=True)

        # Tokenize input dataset
        if self.with_tokenization:
            print(f'Generating basic tokenization for {dataset_name} dataset.')
            dataframe = tokenize_dataset(dataframe, stem=True)

        print('Insert semantic content inside each tuple.')

        # Insert <COL> information
        if self.insert_col:
            cols = dataframe.columns
            new_cols = []

            for col in cols:
                new_col = '__idx_{}'.format(col)
                new_col_metadata = '_'.join(col.split(' '))
                new_col_metadata = 'cid__{}__{}'.format(dataset_name, new_col_metadata)
                dataframe[new_col] = new_col_metadata
                new_cols.append(new_col)
                new_cols.append(col)

            dataframe = dataframe[new_cols]

        # Insert idx__ID information
        dataframe['__idx'] = ['idx__{}__{}'.format(dataset_name, i) for i in range(len(dataframe))]

        # Create sentence list
        print(f'Generating sentences from {dataset_name} dataset.')
        dataframe = dataframe.apply(lambda x: ' '.join(x.dropna().astype(str).to_list()), axis=1)
        sentences_array = dataframe.values

        # Check permutation
        print('Using data augmentation.')
        if self.permutation_rate >= 1:
            sentences_array = sentence_permutation(sentences_array, self.permutation_rate)

        return sentences_array

    def fit(self, df, name='test_name'):
        os.makedirs('pipeline/embeddings', exist_ok=True)
        os.makedirs('pipeline/walks', exist_ok=True)

        print('#' * 80)
        t_start = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Starting run...', t_start))

        # Walks generation
        walks = self.generate_walks(df, name)

        print('Save Walks...')
        walks_name = os.path.join('pipeline/walks/', 'base_{}.walks'.format(name))

        with open(walks_name, 'w') as f:
            for val in walks:
                f.write(val + '\n')

        print('Start embedding {}.'.format(self.training_algorithm))
        base = BaseEmbedding(self.training_algorithm, n_dimensions=self.n_dimensions, window_size=self.window_size)
        base.fit(walks_name)

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
        if not os.path.isdir(input_dir):
            raise ValueError('Input dir does not exists: {}'.format(input_dir))

        os.makedirs('pipeline/embeddings', exist_ok=True)
        os.makedirs('pipeline/walks', exist_ok=True)

        # Get list of tables names
        file_list = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

        print('#' * 80)
        t_start = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Starting run...', t_start))

        dbms_walks = []
        # For each dataset we compute walks
        for filename in file_list:
            name = filename.split('.')[0]

            # Read dataset
            print('Filename: {}'.format(filename))
            df = pd.read_csv(os.path.join(input_dir, filename))

            # Walks generation
            walks = self.generate_walks(df, name)
            dbms_walks.append(walks)
            print('\n')

        print('Save Walks...')
        walks_name = os.path.join('pipeline/walks/base_datasets.walks')

        with open(walks_name, 'w') as f:
            for walk in dbms_walks:
                for val in walk:
                    f.write(val + '\n')

        print('Start embedding {}.'.format(self.training_algorithm))
        base = BaseEmbedding(self.training_algorithm, n_dimensions=self.n_dimensions, window_size=self.window_size)
        base.fit(walks_name)

        name = 'base_{}__datasets.emb'.format(self.training_algorithm)
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

    def load_embedding(self, embedding_file):
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
        keys = np.array([self.keys[i] for i in range(len(self.keys)) if cond[i]])
        new_keys = keys[indexes]

        return new_keys

    def get_best_record(self, sentence, list_records, k=1, distance='cosine', withMean=True):
        cond = [True if x in list_records else False for x in self.keys]
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
        keys = np.array([self.keys[i] for i in range(len(self.keys)) if cond[i]])
        new_keys = keys[indexes]

        return new_keys


if __name__ == '__main__':
    # Add nltk data directory
    add_nltk_path('/home/mirko/nltk_data')
    add_nltk_path('pipeline/nlp/')

    # Check nltk library dependency
    check_nltk_library()

    # Check pipeline directory
    assert os.path.isdir('pipeline'), 'Pipeline directory does not exist'

    args = get_wrapper_arguments()

    # Define model
    wrapper = BaseWrapper()
    if args.dbms:
        # Generate dbms embedding
        print('Start embedding dbms')
        wrapper.fit_db(args.file)
    else:
        # Read input dataset
        input_file = args.file
        print('Read input dataset')
        df = pd.read_csv(input_file)

        file_name = str(os.path.basename(input_file).split('.')[0])

        # Generate embedding
        print('Start embedding dataset')
        wrapper.fit(df, name=file_name)

    print(':)')
