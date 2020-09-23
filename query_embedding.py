import os
import re
import warnings
import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from iteration_utilities import unique_everseen, duplicates
from tables.atom import EnumAtom

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from local_embedding import create_local_embedding


def extract_query(file):
    '''
    File Structure:
    # [reference standard]
    # imdb2009
    # <query>
    <relevant result>
    <relevant result>
    ...
    '''
    f = open(file)
    lines = f.readlines()
    query = lines[2]                            # Take only the third line
    query = query.replace('# ', '')             # Remove the initial '# '
    return query


def extract_tokens(query):
    tokenizer = RegexpTokenizer(r'\w+')         # Word Tokenizer
    tokens = tokenizer.tokenize(query)
    return tokens


def tokens2embeddings(tokens, mat, keys):
    embeddings = []
    for token in tokens:
        tt_token = 'tt__'+token
        if  tt_token in keys:
            idx = keys.index(tt_token)
            embeddings.append(mat[idx])
            print('# Found embedding for token {} at idx {}'.format(tt_token, idx))
        elif token in keys:
            idx = keys.index(token)
            embeddings.append(mat[idx])
            print('# Found embedding for token {} at idx {}'.format(token, idx))
        else:
            print('# No embedding found for token {}'.format(token))
    return embeddings


def compute_similarity(embeddings, mat):
    sims = []
    for embedding in embeddings:
        emb = np.array(embedding)
        emb = emb.reshape(-1, 1)
        emb_transpose = np.transpose(emb)
        similarity = cosine_similarity(emb_transpose, mat)
        sims.append(similarity)
    return sims


def get_max_indices(sim_array, keys):
    counter_rid = 0
    rids_indices = []
    while counter_rid < int(sim_array.shape[0]/10):
        max_idx = np.argmax(sim_array, axis=0)
        if keys[max_idx].startswith("idx__"):
            counter_rid += 1
            rids_indices.append(max_idx)
        sim_array[max_idx] = 0
    return rids_indices


def get_top5RID(similarities, keys):
    if len(similarities) > 1:
        rid_indices = []
        for similarity in similarities:
            sim = similarity.flatten()
            rids = get_max_indices(sim, keys)
            rid_indices += rids
        ranking = list(unique_everseen(duplicates(rid_indices)))
    else:
        sim = similarities[0].flatten()
        rids = get_max_indices(sim, keys)
        ranking = rids
    ranking = ranking[:5]
    return ranking


def create_query_embedding(input_file, mat, keys):
    '''
    input_file --> Dataset in file csv.
    mat --> NxM matrix, where N is the number of tokens that convert and M is 
    the feature vector size (the embedding).
    keys --> List of rows values.
    '''
    filename = os.path.basename(input_file).split('.')[0]
    path = 'pipeline/queries/IMDB/'
    query_dir = sorted(os.listdir(path))
    if len(query_dir) == 0:
        print('# The directory {} is empty!'.format(path))
        return -1
    
    df = pd.read_csv(input_file)
    output_df = pd.DataFrame(columns=('File', 'Query', 'Pos', 'RID', 'Name', 'SearchID'))

    for file in query_dir:
        if re.match('^\d+', file):
            path_file = path + file
            print('# Query extraction from {}'.format(file))
            query = extract_query(path_file)
            print('# Query extracted: {}'.format(query), end='')
            tokens = extract_tokens(query)
            print('# Tokens extracted: {}'.format(tokens))
            print('# Embeddings extraction...')
            embeddings = tokens2embeddings(tokens, mat, keys)
            if len(embeddings) == 0:
                print('# No embeddings found for query --> {}'.format(query))
            else:
                print('# Computing similarities...')
                similarities = compute_similarity(embeddings, mat)
                print('# Searching the top5 similar RIDs...')
                top5rids = get_top5RID(similarities, keys)
                for i, idx in enumerate(top5rids):
                    index = int(keys[idx].replace('idx__', ''))
                    name = df.iloc[index-2]['name']
                    searchID = df.iloc[index-2]['__search_id']
                    print(f'{i+1}) --> {keys[idx]} --> {name} --> {searchID}')
                    q = ' '.join(tokens)
                    values_to_add = {'File': file, 'Query': q, 'Pos': i+1, 
                                    'RID': keys[idx], 'Name': name, 
                                    'SearchID': searchID}
                    row_to_add = pd.Series(values_to_add)
                    output_df = output_df.append(row_to_add, ignore_index=True)
                print()
    output_df.to_csv('pipeline/debuggings/output_{}.csv'.format(filename))
    return 0


if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    mat, keys = create_local_embedding(input_file)
    if create_query_embedding(input_file, mat, keys) == -1:
        print('Ops! The function failed!')
