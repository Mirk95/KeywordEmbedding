import os
import re
import nltk
import warnings
from ast import literal_eval

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from wrapper.retro_numeric_wrapper import RETRONumericWrapper


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

    input_dir = 'pipeline/datasets/'
    emb_file = 'pipeline/output_retro_numeric/retrofitted_vectors.wv'

    # Define embedding model
    wrapper = RETRONumericWrapper(n_iterations=10,
                                  alpha=1.0,
                                  beta=0.0,
                                  gamma=3.0,
                                  delta=1.0,
                                  number_dims=300,
                                  standard_deviation=1.0,
                                  table_blacklist=[],
                                  with_tokenization=True,)

    # Load embedding matrix
    wrapper.load_embedding(emb_file)
    wrapper.create_idx_embeddings(input_dir)
    wrapper.transform_keys_embeddings()

    label_dir = 'pipeline/queries/IMDB'
    label_files = ["{:03d}.txt".format(x) for x in range(1, 11)]

    for label_name in label_files:
        print('\n File: {}'.format(label_name))
        keywords, search_ids = extract_keywords(os.path.join(label_dir, label_name))

        print('Keywords {}'.format(keywords))
        print('Label search id {}'.format(search_ids))

        print('\n Embedding extraction')

        # Preprocess sentence with trained model preprocessing
        sentence = wrapper.preprocess_sentence(keywords)

        # Get nearest_neighbour
        print('\n Get K nearest records')
        neighbours = wrapper.get_k_nearest_token(sentence, k=5, distance='cosine', pref='idx', withMean=False)
        neighbours = [int(x.replace('idx__', '')) for x in neighbours]
        # print(df.loc[neighbours, ['name', '__search_id']])

        # validate results

    # Keyword Search
    # if create_query_embedding(input_file, mat, keys, mean_emb=False) == -1:
    # print('Ops! The function failed!')
