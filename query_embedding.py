import os
import re
import warnings
from ast import literal_eval

import nltk
import pandas as pd


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from wrapper.embdi_wrapper import EmbDIWrapper


def extract_query(sentence):
    pattern = r'(\w+)'
    query = re.findall(pattern, sentence)
    query = ' '.join(query)
    return query


def extract_gt(search_ids):
    gt_list = []
    for line in search_ids:
        line = literal_eval(line)
        gt_list.append(line)
    return gt_list


def extract_keywords(filename):

    #    File Structure:
    #    # [reference standard]
    #    # imdb2009
    #    # <query>
    #    <relevant result>
    #    <relevant result>

    with open(filename, 'r') as f:
        lines = f.readlines()

    sentence = extract_query(lines[2])
    search_ids = extract_gt(lines[3:])

    return sentence, search_ids


if __name__ == '__main__':
    # Add nltk data directory
    nltk.data.path.append('/Users/francesco/Development')

    input_file = 'pipeline/datasets/name.csv'
    emb_file = 'pipeline/embeddings/name.emb'

    name = str(os.path.basename(input_file).split('.')[0])

    # read input dataset
    df = pd.read_csv(input_file, quotechar='"', error_bad_lines=False)
    df = df.head(100)

    # define embedding model
    wrapper = EmbDIWrapper(n_dimensions=300,
                           window_size=3,
                           n_sentences='default',
                           training_algorithm='word2vec',
                           learning_method='skipgram',
                           with_tokenization=True)

    # load embedding matrix
    # wrapper.fit(df, name=name)
    wrapper.load_emebedding(emb_file)

    label_dir = 'pipeline/queries/IMDB'
    label_files = ["{:03d}.txt".format(x) for x in range(1, 11)]

    for label_name in label_files:
        print('\n File: {}'.format(label_name))
        keywords, search_ids = extract_keywords(os.path.join(label_dir, label_name))

        print('Keywords {}'.format(keywords))
        print('Label search id {}'.format(search_ids))

        print('\n Embedding extraction')

        # preprocess sentence with trained model preprocessing
        sentence = wrapper.preprocess_sentence(keywords)

        # get nearest_neighbour
        print('\n Get K nearest records')
        neighbours = wrapper.get_k_nearest_token(sentence, k=5, distance='cosine', pref='idx', withMean=False)
        neighbours = [int(x.replace('idx__', '')) for x in neighbours]
        print(df.loc[neighbours, ['name', '__search_id']])

        # validate results

    # Keyword Search
    # if create_query_embedding(input_file, mat, keys, mean_emb=False) == -1:
    # print('Ops! The function failed!')
