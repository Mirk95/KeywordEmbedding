import os
import re
import datetime
import warnings
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer

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
    print('# Query extraction from {}'.format(file))
    f = open(file)
    lines = f.readlines()
    query = lines[2]                            # Take only the third line
    query = query.replace('# ', '')             # Remove the initial '# '
    print('# Query extracted: {}'.format(query), end='')
    return query


def extract_tokens(df, query):
    tokenizer = RegexpTokenizer(r'\w+')         # Word Tokenizer
    tokens = tokenizer.tokenize(query)
    df_columns = df.columns
    # Remove the names of df columns from the list of tokens
    list_tokens = [t for t in tokens if t not in df_columns]
    print('# Tokens extracted: {}'.format(list_tokens))
    return list_tokens


def tokens2embedding(tokens, mat, keys):
    print ('# Embeddings extraction from matrix mat')
    tokens = [t.capitalize() for t in tokens]
    embeddings = []
    for token in tokens:
        if token in keys:
            idx = keys.index(token)
            embeddings.append(mat[idx])
    
    tokens_tt = ['tt__'+t for t in tokens]
    for token_tt in tokens_tt:
        if token_tt in keys:
            idx = keys.index(token_tt)
            embeddings.append(mat[idx])

    emb = np.array(embeddings)
    if emb.shape[0] > 1:
        # Apply Mean
        mean_vec = np.zeros((emb.shape[1]))
        for j in range(emb.shape[1]):
            mean = 0.00
            sum = 0.00
            for i in range(emb.shape[0]):
                sum += mat[i][j]
            mean = float(sum / emb.shape[0])
            mean_vec[j] = mean
        return mean_vec
    else:
        return emb


def compute_similarity(embedding, mat, keys):
    pass


def create_query_embedding(input_file, mat, keys):
    '''
    input_file --> Dataset in fiembeddingsle csv.
    mat --> NxM matrix, where N is the number of tokens that convert and M is 
    the feature vector size (the embedding).
    keys --> List of rows values.
    '''
    df = pd.read_csv(input_file)
    path = 'pipeline/queries/IMDB/'
    query_dir = sorted(os.listdir(path))
    if len(query_dir) == 0:
        print('# The directory {} is empty!'.format(path))
        return -1

    for file in query_dir:
        if re.match('^\d+', file):
            path_file = path + file
            query = extract_query(path_file)
            tokens = extract_tokens(df, query)
            embedding = tokens2embedding(tokens, mat, keys)
            if embedding.size == 0:
                print('# No embeddings found for query {}'.format(query))
                print()
            else:
                print('# Ok, looking for similar embeddings...')
                compute_similarity(embedding, mat, keys)
        else:
            print('# No file found in the directory starting with a number!')
            return -1


if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    mat, keys = create_local_embedding(input_file)
    if create_query_embedding(input_file, mat, keys) == -1:
        print('Ops! The function failed!')
