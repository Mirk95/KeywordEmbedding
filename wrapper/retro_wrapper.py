import os
import csv
from gensim.models.keyedvectors import KeyedVectors

from RETRO.graph_generation import main as graph_generation
from RETRO.group_extraction import main as group_extraction
from RETRO.matrix_retrofit import main as retrofit
from RETRO.gml2json import main as gml2json


class RETROWrapper(object):
    def __init__(self):
        pass

    def fit_db(self):
        configuration = {
            "WE_ORIGINAL_TABLE_NAME": "google_vecs",
            "WE_ORIGINAL_TABLE_PATH": "pipeline/vectors/google_vecs.csv",
            "RETRO_TABLE_CONFS": ["config/retro_vecs.config"],
            "DATASETS_PATH": "pipeline/datasets/",
            "SCHEMAS_PATH": "pipeline/schemas/",
            "OUTPUT_PATH": "pipeline/output/",
            "SCHEMA_GRAPH_PATH": "pipeline/output/schema.gml",
            "SCHEMA_JSON_GRAPH_PATH": "pipeline/output/schema.json",
            "TABLE_BLACKLIST": [],
            "COLUMN_BLACKLIST": [],
            "RELATION_BLACKLIST": [],
            "VECTORS_LOCATION": "pipeline/vectors/GoogleNews-vectors-negative300.bin",
            "OUTPUT_LOCATION": "pipeline/vectors/google_vecs.csv",
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

        os.makedirs(configuration['OUTPUT_PATH'], exist_ok=True)
        if not os.path.exists(configuration['OUTPUT_LOCATION']):
            word_vectors = KeyedVectors.load_word2vec_format(configuration['VECTORS_LOCATION'], binary=True)
            with open(configuration['OUTPUT_LOCATION'], 'w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['word'] + ['vector'])
                words = word_vectors.vocab.keys()
                for word in words:
                    vector = word_vectors.get_vector(word).tolist()
                    vector = str(vector).replace(',', '')
                    row = [word] + [vector]
                    writer.writerow(row)

        print('Create schema graph in', configuration['SCHEMA_GRAPH_PATH'], '...')
        graph_generation(configuration)
        gml2json(configuration)  # only for visualization
        print('Extract groups and generate', configuration['GROUPS_FILE_NAME'], '...')
        group_extraction(configuration)
        print('Start retrofitting ...')
        retrofit(configuration)


if __name__ == '__main__':
    # define model
    wrapper = RETROWrapper()
    wrapper.fit_db()
    print(':)')
