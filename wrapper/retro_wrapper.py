import os
import csv
import requests
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


class RETROWrapper(object):
    def __init__(self):
        # embedding values
        self.mat = np.array([])
        self.keys = []

    def fit_db(self):
        configuration = {
            "VECTORS_PATH": "pipeline/vectors",
            "DATASETS_PATH": "pipeline/datasets/",
            "SCHEMAS_PATH": "pipeline/schemas/",
            "OUTPUT_PATH": "pipeline/output/",
            "VECTORS_LOCATION": "pipeline/vectors/GoogleNews-vectors-negative300.bin.gz",
            "WE_ORIGINAL_TABLE_NAME": "google_vecs",
            "WE_ORIGINAL_TABLE_PATH": "pipeline/vectors/google_vecs.csv",
            "SCHEMA_GRAPH_PATH": "pipeline/output/schema.gml",
            "SCHEMA_JSON_GRAPH_PATH": "pipeline/output/schema.json",
            "TABLE_BLACKLIST": [],
            "COLUMN_BLACKLIST": [],
            "RELATION_BLACKLIST": [],
            "GROUPS_FILE_NAME": "pipeline/output/groups.pickle",
            "ITERATIONS": 10,
            "RETRO_VECS_FILE_NAME": "pipeline/output/retrofitted_vectors.wv",
            "TOKENIZATION": "simple",
            "ALPHA": 1.0,
            "BETA": 0.0,
            "GAMMA": 3.0,
            "DELTA": 3.0,
            "MAX_ROWS": 100000
        }

        os.makedirs(configuration['VECTORS_PATH'], exist_ok=True)
        os.makedirs(configuration['OUTPUT_PATH'], exist_ok=True)

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


if __name__ == '__main__':
    # Add nltk data directory
    add_nltk_path('/home/mirko/nltk_data')
    add_nltk_path('pipeline/nlp/')

    # check nltk library dependency
    check_nltk_library()

    # define model
    wrapper = RETROWrapper()
    wrapper.fit_db()
    print(':)')
