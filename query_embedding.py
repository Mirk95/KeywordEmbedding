import os
import re
import nltk
import argparse
import warnings
import pandas as pd
from ast import literal_eval

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from wrapper.embdi_wrapper import EmbDIWrapper
    from wrapper.retro_numeric_wrapper import RETRONumericWrapper
    from wrapper.base_wrapper import BaseWrapper


def get_arguments():
    parser = argparse.ArgumentParser(description='Run Query Embedding')
    parser.add_argument('--wrapper',
                        type=str,
                        required=True,
                        choices=['base', 'embdi', 'retro'],
                        help='Wrapper to use for query embedding')
    args = parser.parse_args()
    return args


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
    nltk.data.path.append('/home/mirko/nltk_data')

    # Retrieve Arguments
    args = get_arguments()

    # Define embedding model
    if args.wrapper == 'embdi':
        wrapper = EmbDIWrapper(n_dimensions=300,
                               window_size=3,
                               n_sentences='default',
                               training_algorithm='word2vec',
                               learning_method='skipgram',
                               with_tokenization=True)
    elif args.wrapper == 'retro':
        wrapper = RETRONumericWrapper(n_iterations=10, alpha=1.0, beta=0.0,
                                      gamma=3.0, delta=1.0, number_dims=300,
                                      standard_deviation=1.0,
                                      table_blacklist=[],
                                      with_tokenization=True)
    else:
        # 'base' Wrapper
        wrapper = BaseWrapper(training_algorithm='word2vec_CBOW',
                              n_dimensions=300, window_size=3,
                              with_tokenization=True, ignore_columns=None,
                              insert_col=True, permutation_rate=10)

    emb_file = 'pipeline/embeddings/' + args.wrapper + '__datasets.emb'
    if os.path.isfile(emb_file):
        # Load embedding matrix
        wrapper.load_embedding(emb_file)

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
            for neighbour in neighbours:
                table, idx = neighbour.split('__')
                df = pd.read_csv('pipeline/datasets/' + table + '.csv', na_filter=False)
                print(df.loc[neighbour, ['__search_id']])
    else:
        raise ValueError(f'The file {emb_file} does not exist!')
