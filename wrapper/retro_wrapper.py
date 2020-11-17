import os
import csv
import requests
import datetime
import warnings
import numpy as np
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors

from preprocessing.utils import check_nltk_library, add_nltk_path, get_wrapper_arguments

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from RETRO.graph_generation import main as graph_generation
    from RETRO.group_extraction import main as group_extraction
    from RETRO.matrix_retrofit import main as retrofit
    from RETRO.gml2json import main as gml2json

OUTPUT_FORMAT = '# {:.<60} {}'


def download_file(url: str, vectors_path: str):
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


class RETROWrapper(object):
    def __init__(self,
                 n_iterations=10,
                 alpha=1.0,
                 beta=0.0,
                 gamma=3.0,
                 delta=3.0,
                 tokenization='simple',
                 table_blacklist=[],
                 ):

        # embedding values
        self.mat = np.array([])
        self.keys = []

        # embedding model parameters
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.tokenization = tokenization
        self.table_blacklist = table_blacklist

    def fit(self):
        # configuration dictionary
        configuration = {
            "DATASETS_PATH": "pipeline/datasets/",
            "VECTORS_PATH": "pipeline/vectors/",
            "SCHEMAS_PATH": "pipeline/schemas/",
            "COLUMNS_TYPE_PATH": "pipeline/columns/",
            "OUTPUT_PATH": "pipeline/output/",
            "VECTORS_LOCATION": "pipeline/vectors/GoogleNews-vectors-negative300.bin.gz",
            "WE_ORIGINAL_TABLE_NAME": "google_vecs",
            "WE_ORIGINAL_TABLE_PATH": "pipeline/vectors/google_vecs.csv",
            "SCHEMA_GRAPH_PATH": "pipeline/output/schema.gml",
            "SCHEMA_JSON_GRAPH_PATH": "pipeline/output/schema.json",
            "GROUPS_FILE_NAME": "pipeline/output/groups.pk",
            "RETRO_VECS_FILE_NAME": "pipeline/output/retrofitted_vectors.wv",
            "TABLE_BLACKLIST": self.table_blacklist,
            "COLUMN_BLACKLIST": [],
            "RELATION_BLACKLIST": [],
            "CREATE_NEW_COLUMN_INDEX": False,
            "ITERATIONS": self.n_iterations,
            "TOKENIZATION": self.tokenization,
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
        gml2json(configuration)  # only for visualization
        print('Extract groups and generate', configuration['GROUPS_FILE_NAME'], '...')
        group_extraction(configuration)
        print('Start retrofitting ...')
        retrofit(configuration)

        self.mat, self.keys = prepare_emb_matrix(configuration['RETRO_VECS_FILE_NAME'])

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
            "OUTPUT_PATH": "pipeline/output/",
            "VECTORS_LOCATION": "pipeline/vectors/GoogleNews-vectors-negative300.bin.gz",
            "WE_ORIGINAL_TABLE_NAME": "google_vecs",
            "WE_ORIGINAL_TABLE_PATH": "pipeline/vectors/google_vecs.csv",
            "SCHEMA_GRAPH_PATH": "pipeline/output/schema.gml",
            "SCHEMA_JSON_GRAPH_PATH": "pipeline/output/schema.json",
            "GROUPS_FILE_NAME": "pipeline/output/groups.pk",
            "RETRO_VECS_FILE_NAME": "pipeline/output/retrofitted_vectors.wv",
            "TABLE_BLACKLIST": ['char_name', 'movie_info', 'role_type', 'title'],
            "COLUMN_BLACKLIST": [],
            "RELATION_BLACKLIST": [],
            "CREATE_NEW_COLUMN_INDEX": False,
            "ITERATIONS": self.n_iterations,
            "TOKENIZATION": self.tokenization,
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
        gml2json(configuration)  # only for visualization
        print('Extract groups and generate', configuration['GROUPS_FILE_NAME'], '...')
        group_extraction(configuration)
        print('Start retrofitting ...')
        retrofit(configuration)

        self.mat, self.keys = prepare_emb_matrix(configuration['RETRO_VECS_FILE_NAME'])

        t_end = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Ending run.', t_end))
        dt = t_end - t_start
        print('# Time required: {}'.format(dt.total_seconds()))

        print('')
        print('Embedding Matrix shape: {}'.format(self.mat.shape))
        print('Keys number: {}'.format(len(self.keys)))


if __name__ == '__main__':
    # Add nltk data directory
    add_nltk_path('/home/mirko/nltk_data')
    add_nltk_path('pipeline/nlp/')

    # Check nltk library dependency
    check_nltk_library()

    args = get_wrapper_arguments()

    # Define model
    wrapper = RETROWrapper()
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
        wrapper.fit()

    print(':)')
