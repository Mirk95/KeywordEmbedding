import os
import re
import datetime
import warnings

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
    query = lines[2]                        # Take only the third line
    query = query.replace('# ', '')         # Remove the initial '# '
    print('# Query extracted: {}'.format(query))
    return query


def extract_tokens(query):
    print('# Tokens extraction from query {}'.format(query))
    tokenizer = RegexpTokenizer(r'\w+')
    list_tokens = tokenizer.tokenize(query)
    print('# Tokens extracted: {}'.format(list_tokens))
    return list_tokens


def create_embeddings(tokens, mat, keys):
    pass


def create_query_embedding(mat, keys):
    '''
    mat --> NxM matrix, where N is the number of tokens that convert and M is 
    the feature vector size (the embedding).
    keys --> list of rows values
    '''
    path = 'pipeline/queries/IMDB/'
    query_dir = sorted(os.listdir(path))
    if len(query_dir) == 0:
        print('# The directory {} is empty!'.format(path))
        return -1

    for file in query_dir:
        if re.match('^\d+', file):
            path_file = path + file
            query = extract_query(path_file)
            tokens = extract_tokens(query)
            if tokens and len(tokens) < 10:
                create_embeddings(tokens, mat, keys)



if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    mat, keys = create_local_embedding(input_file)
    create_query_embedding(mat, keys)
