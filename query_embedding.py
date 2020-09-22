import os
import re
import warnings
import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity

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


def generate_tokens_combination(tokens):
    final_tokens = []
    for t in tokens:
        final_tokens.append(t)
        final_tokens.append(t+',')
        final_tokens.append('tt__'+t)
        final_tokens.append('tt__'+t+',')
    
    tokens_capitalized = [t.capitalize() for t in tokens]
    for tc in tokens_capitalized:
        final_tokens.append(tc)
        final_tokens.append(tc+',')
        final_tokens.append('tt__'+tc)
        final_tokens.append('tt__'+tc+',')

    return final_tokens


def tokens2embedding(tokens, mat, keys):
    embeddings = []
    final_tokens = generate_tokens_combination(tokens)
    for token in final_tokens:
        indices = [i for i, el in enumerate(keys) if el==token]
        for idx in indices:
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


def compute_similarity(embedding, mat):
    embedding = embedding.reshape(-1, 1)
    emb_transpose = np.transpose(embedding)
    similarity = cosine_similarity(emb_transpose, mat)
    return similarity


def get_top5_indices(sim):
    indices = np.argpartition(sim, -5)[-5:]
    indices = indices[np.argsort(-sim[indices])]
    return indices.tolist()


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
    
    output_df = pd.DataFrame(columns=('File', 'Query', 'Pos', 'Sim', 'RID'))

    for file in query_dir:
        if re.match('^\d+', file):
            path_file = path + file
            print('# Query extraction from {}'.format(file))
            query = extract_query(path_file)
            print('# Query extracted: {}'.format(query), end='')
            tokens = extract_tokens(query)
            print('# Tokens extracted: {}'.format(tokens))
            print ('# Embeddings extraction from matrix mat')
            embedding = tokens2embedding(tokens, mat, keys)
            if embedding.size == 0:
                print('# No embeddings found for query --> {}'.format(query))
            else:
                print('# Computing similarity vector')
                sim = compute_similarity(embedding, mat)
                print('# Getting the Top5 indices')
                sim = sim.flatten()
                idxs = get_top5_indices(sim)
                for i, idx in enumerate(idxs):
                    print('{}) -->     {}'.format(i+1, keys[idx]))
                    q = ' '.join(tokens)
                    values_to_add = {'File': file, 'Query': q, 'Pos': i+1, 
                                    'Sim': sim[idx], 'RID': keys[idx]}
                    row_to_add = pd.Series(values_to_add)
                    output_df = output_df.append(row_to_add, ignore_index=True)
                print()
    output_df.to_csv('pipeline/debuggings/output_{}.csv'.format(filename))


if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    mat, keys = create_local_embedding(input_file)
    if create_query_embedding(input_file, mat, keys) == -1:
        print('Ops! The function failed!')
