import os
import csv
import requests
import datetime
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import linesep
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from preprocessing.tokenizer import tokenize_sentence
from preprocessing.utils import check_nltk_library, add_nltk_path, get_wrapper_arguments

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from RETRO_Numeric.graph_generation import main as graph_generation
    from RETRO_Numeric.group_extraction import main as group_extraction
    from RETRO_Numeric.matrix_retrofit import main as retrofit
    from RETRO_Numeric.gml2json import main as gml2json
    from RETRO_Numeric.retro_utils import tokenize

OUTPUT_FORMAT = '# {:.<60} {}'


def download_file(url, vectors_path):
    print(f"Beginning {vectors_path} file download...")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(vectors_path, 'wb') as file, tqdm(
        desc=vectors_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    print("Download complete!")


def transform_vecs(vectors_location, output_location):
    print(f"Beginning {vectors_location} file transformation into csv...")
    word_vectors = KeyedVectors.load_word2vec_format(vectors_location, binary=True)
    with open(output_location, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['word'] + ['vector'])
        words = word_vectors.vocab.keys()
        for word in words:
            vector = word_vectors.get_vector(word).tolist()
            vector = str(vector).replace(',', '')
            row = [word] + [vector]
            writer.writerow(row)
    print("Transformation complete!")


def create_idx_embeddings(keys, mat, datasets_directory):
    for filename in os.listdir(datasets_directory):
        name = filename.split('.')[0]
        check_emb = [i for i in keys if i.startswith(name)]
        if len(check_emb) != 0:
            df = pd.read_csv(datasets_directory + filename, na_filter=False)
            columns = df.columns.tolist()
            for idx, row in df.iterrows():
                embeddings = []
                for col in columns:
                    value = tokenize(str(row[col]))
                    for val in value.split('_'):
                        key = name + '.' + col + '#' + str(val)
                        if key in keys:
                            # Found Embedding
                            index_key = keys.index(key)
                            embeddings.append(mat[index_key])
                        else:
                            # Try with None:
                            key_with_none = name + '.' + col + '#' + str(None)
                            if key_with_none in keys:
                                # Found Embedding
                                index_key = keys.index(key_with_none)
                                embeddings.append(mat[index_key])
                if len(embeddings) > 1:
                    new_embedding_idx = name + '.idx#' + str(idx)
                    new_embedding = np.mean(embeddings, axis=0, keepdims=True)
                    keys.append(new_embedding_idx)
                    mat = np.append(mat, new_embedding, axis=0)
    return keys, mat


def transform_keys_embeddings(keys):
    new_keys = []
    for key in keys:
        table_name = key.split('.')[0]
        column_name = key.split('.')[1].split('#')[0]
        value = key.split('#')[1]
        if column_name == 'idx':
            new_key = 'idx__' + str(table_name) + '__' + str(value)
        else:
            if value == '':
                new_key = 'cid__' + str(table_name) + '__' + str(column_name)
            else:
                new_key = str(value)
        new_keys.append(new_key)

    assert [len(keys) != len(new_keys)], "Error, something went wrong during keys transformation!"
    return new_keys


def output_vectors(term_list, Mk, output_file, datasets_path, with_zero_vectors=True):
    keys, matrix = create_idx_embeddings(term_list, Mk, datasets_path)
    keys_transformed = transform_keys_embeddings(keys)
    # Init output file
    f_out = open(output_file, 'w')
    if with_zero_vectors:
        # Write meta information
        f_out.write('%d %d' % (matrix.shape[0], matrix.shape[1]) + linesep)
        # Write term vector pairs
        for i, term in enumerate(keys_transformed):
            if i % 1000 == 0:
                print('Exported', i, 'term vectors | Current term:', term)
            f_out.write('%s %s' % (term, ' '.join([str(x) for x in matrix[i]])))
            f_out.write(linesep)
    else:
        counter = 0
        for i, term in enumerate(keys_transformed):
            is_all_zero = np.all((matrix[i] == 0))
            if not is_all_zero:
                counter += 1

        # Init output file
        f_out = open(output_file, 'w')
        # Write meta information
        f_out.write('%d %d' % (counter, matrix.shape[1]) + linesep)
        # Write term vector pairs
        for i, term in enumerate(keys_transformed):
            is_all_zero = np.all((matrix[i] == 0))
            if not is_all_zero:
                print('Exported', i, 'term vectors | Current term:', term)
                f_out.write('%s %s' % (term, ' '.join([str(x) for x in matrix[i]])))
                f_out.write(linesep)
    f_out.close()
    return


def prepare_emb_matrix(embeddings_file):
    # Reading the reduced file
    keys = []
    with open(embeddings_file, 'r') as fp:
        lines = fp.readlines()
        sizes = lines[0].split()
        sizes = [int(_) for _ in sizes]
        mat = np.zeros(shape=sizes)
        for n, line in enumerate(lines[1:]):
            ll = line.strip().split()
            mat[n, :] = np.array(ll[1:])
            keys.append(ll[0])
    return mat, keys


class RETRONumericWrapper(object):
    def __init__(self,
                 n_iterations=10,
                 alpha=1.0,
                 beta=0.0,
                 gamma=3.0,
                 delta=1.0,
                 number_dims=300,
                 standard_deviation=1.0,
                 table_blacklist=[],
                 with_tokenization=True,
                 ):

        # Embedding values
        self.mat = np.array([])
        self.keys = []

        # Embedding model parameters
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.number_dims = number_dims
        self.standard_deviation = standard_deviation
        self.table_blacklist = table_blacklist

        # Preprocessing data values
        self.with_tokenization = with_tokenization

    def fit(self, input_file):
        # Configuration dictionary
        configuration = {
            "DATASETS_PATH": "pipeline/datasets/",
            "VECTORS_PATH": "pipeline/vectors/",
            "SCHEMAS_PATH": "pipeline/schemas/",
            "COLUMNS_TYPE_PATH": "pipeline/columns/",
            "OUTPUT_PATH": "pipeline/embeddings/",
            "VECTORS_LOCATION": "pipeline/vectors/GoogleNews-vectors-negative300.bin.gz",
            "WE_ORIGINAL_TABLE_NAME": "google_vecs",
            "WE_ORIGINAL_TABLE_PATH": "pipeline/vectors/google_vecs.csv",
            "SCHEMA_GRAPH_PATH": "pipeline/embeddings/schema.gml",
            "SCHEMA_JSON_GRAPH_PATH": "pipeline/embeddings/schema.json",
            "GROUPS_FILE_NAME": "pipeline/embeddings/groups.pk",
            "TABLE_BLACKLIST": self.table_blacklist,
            "COLUMN_BLACKLIST": [],
            "RELATION_BLACKLIST": [],
            "M0_ZERO_COLUMNS": [],
            "ITERATIONS": self.n_iterations,
            "TOKENIZATION_SETTINGS": {
                "TEXT_TOKENIZATION": "simple",
                "NUMERIC_TOKENIZATION": {
                    "MODE": "unary-random-dim",
                    "BUCKETS": True,
                    "NUMBER_DIMS": self.number_dims,
                    "STANDARD_DEVIATION": self.standard_deviation,
                    "NORMALIZATION": True
                }
            },
            "ALPHA": self.alpha,
            "BETA": self.beta,
            "GAMMA": self.gamma,
            "DELTA": self.delta
        }

        # RETRO directory
        os.makedirs(configuration['VECTORS_PATH'], exist_ok=True)
        os.makedirs(configuration['OUTPUT_PATH'], exist_ok=True)

        print('#' * 80)
        t_start = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Starting run.', t_start))

        if not os.path.exists(configuration['VECTORS_LOCATION']):
            # Download the GoogleNews Vectors File
            url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
            download_file(url, configuration['VECTORS_LOCATION'])

        if not os.path.exists(configuration['WE_ORIGINAL_TABLE_PATH']):
            # Saving the binary file in csv
            transform_vecs(configuration['VECTORS_LOCATION'], configuration['WE_ORIGINAL_TABLE_PATH'])

        print('Create schema graph in', configuration['SCHEMA_GRAPH_PATH'], '...')
        graph_generation(configuration)
        gml2json(configuration)     # only for visualization
        print('Extract groups and generate', configuration['GROUPS_FILE_NAME'], '...')
        group_extraction(configuration)
        print('Start retrofitting ...')
        term_list, Mk = retrofit(configuration)

        # Output result to file
        output_filename = configuration['OUTPUT_PATH'] + 'retro__' + input_file + '.emb'
        output_vectors(term_list, Mk, output_filename, configuration['DATASETS_PATH'], with_zero_vectors=True)
        print('Exported vectors')

        print("Finished retrofitting for:")
        print("TEXT MODE: ", configuration['TOKENIZATION_SETTINGS']['TEXT_TOKENIZATION'])
        print("NUMERIC MODE: ", configuration['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['MODE'])
        print("\t BUCKETS: ", configuration['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['BUCKETS'])
        print("\t NUMBER DIMS: ", configuration['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['NUMBER_DIMS'])
        print("DELTA: ", configuration['DELTA'])

        self.mat, self.keys = prepare_emb_matrix(output_filename)

        # Remove Schema Graphs and Groups file, to free some memory...
        os.remove(configuration['SCHEMA_GRAPH_PATH'])
        os.remove(configuration['SCHEMA_JSON_GRAPH_PATH'])
        os.remove(configuration['GROUPS_FILE_NAME'])

        t_end = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Ending run.', t_end))
        dt = t_end - t_start
        print('# Time required: {}'.format(dt.total_seconds()))

        print('')
        print('Embedding Matrix shape: {}'.format(self.mat.shape))
        print('Keys number: {}'.format(len(self.keys)))

    def fit_db(self, input_dir):
        if not os.path.isdir(input_dir):
            raise ValueError('Input dir does not exists: {}'.format(input_dir))

        # configuration dictionary
        configuration = {
            "DATASETS_PATH": "pipeline/datasets/",
            "VECTORS_PATH": "pipeline/vectors/",
            "SCHEMAS_PATH": "pipeline/schemas/",
            "COLUMNS_TYPE_PATH": "pipeline/columns/",
            "OUTPUT_PATH": "pipeline/embeddings/",
            "VECTORS_LOCATION": "pipeline/vectors/GoogleNews-vectors-negative300.bin.gz",
            "WE_ORIGINAL_TABLE_NAME": "google_vecs",
            "WE_ORIGINAL_TABLE_PATH": "pipeline/vectors/google_vecs.csv",
            "SCHEMA_GRAPH_PATH": "pipeline/embeddings/schema.gml",
            "SCHEMA_JSON_GRAPH_PATH": "pipeline/embeddings/schema.json",
            "GROUPS_FILE_NAME": "pipeline/embeddings/groups.pk",
            "TABLE_BLACKLIST": self.table_blacklist,
            "COLUMN_BLACKLIST": [],
            "RELATION_BLACKLIST": [],
            "M0_ZERO_COLUMNS": [],
            "ITERATIONS": self.n_iterations,
            "TOKENIZATION_SETTINGS": {
                "TEXT_TOKENIZATION": "simple",
                "NUMERIC_TOKENIZATION": {
                    "MODE": "unary-random-dim",
                    "BUCKETS": True,
                    "NUMBER_DIMS": self.number_dims,
                    "STANDARD_DEVIATION": self.standard_deviation,
                    "NORMALIZATION": True
                }
            },
            "ALPHA": self.alpha,
            "BETA": self.beta,
            "GAMMA": self.gamma,
            "DELTA": self.delta
        }

        # RETRO directory
        os.makedirs(configuration['VECTORS_PATH'], exist_ok=True)
        os.makedirs(configuration['OUTPUT_PATH'], exist_ok=True)

        print('#' * 80)
        t_start = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Starting run.', t_start))

        if not os.path.exists(configuration['VECTORS_LOCATION']):
            # Download the GoogleNews Vectors File
            url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
            download_file(url, configuration['VECTORS_LOCATION'])

        if not os.path.exists(configuration['WE_ORIGINAL_TABLE_PATH']):
            # Saving the binary file in csv
            transform_vecs(configuration['VECTORS_LOCATION'], configuration['WE_ORIGINAL_TABLE_PATH'])

        print('Create schema graph in', configuration['SCHEMA_GRAPH_PATH'], '...')
        graph_generation(configuration)
        gml2json(configuration)     # only for visualization
        print('Extract groups and generate', configuration['GROUPS_FILE_NAME'], '...')
        group_extraction(configuration)
        print('Start retrofitting ...')
        term_list, Mk = retrofit(configuration)

        # Output result to file
        output_filename = configuration['OUTPUT_PATH'] + 'retro__datasets.emb'
        output_vectors(term_list, Mk, output_filename, configuration['DATASETS_PATH'], with_zero_vectors=True)
        print('Exported vectors')

        print("Finished retrofitting for:")
        print("TEXT MODE: ", configuration['TOKENIZATION_SETTINGS']['TEXT_TOKENIZATION'])
        print("NUMERIC MODE: ", configuration['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['MODE'])
        print("\t BUCKETS: ", configuration['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['BUCKETS'])
        print("\t NUMBER DIMS: ", configuration['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['NUMBER_DIMS'])
        print("DELTA: ", configuration['DELTA'])

        self.mat, self.keys = prepare_emb_matrix(output_filename)

        # Remove Schema Graphs and Groups file, to free some memory...
        os.remove(configuration['SCHEMA_GRAPH_PATH'])
        os.remove(configuration['SCHEMA_JSON_GRAPH_PATH'])
        os.remove(configuration['GROUPS_FILE_NAME'])

        t_end = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Ending run.', t_end))
        dt = t_end - t_start
        print('# Time required: {}'.format(dt.total_seconds()))

        print('')
        print('Embedding Matrix shape: {}'.format(self.mat.shape))
        print('Keys number: {}'.format(len(self.keys)))

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

    args = get_wrapper_arguments()

    # Define model
    wrapper = RETRONumericWrapper()
    if args.dbms:
        # Generate dbms embedding
        wrapper.fit_db(args.file)
    else:
        input_file = args.file
        dir_name = os.path.dirname(input_file)

        # Get list of table name
        file_list = [f for f in os.listdir(dir_name) if f.endswith('.csv')]
        file_list.remove(os.path.basename(input_file))
        wrapper.table_blacklist = [file.split('.')[0] for file in file_list]

        # Generate embedding
        file_name = os.path.basename(input_file).split('.')[0]
        wrapper.fit(file_name)

    print(':)')
