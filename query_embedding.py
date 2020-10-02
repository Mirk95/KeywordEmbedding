import os
import re
import warnings
import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from wrapper.embdi_wrapper import create_local_embedding


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


def extract_gt(file):
    f = open(file)
    lines = f.readlines()
    ground_truth = lines[3:]
    gt_list = []
    for line in ground_truth:
        line = line.split("#", 1)[0]            # Remove comment in line
        line = re.sub("[()]", "", line)         # Remove parenthesis () in gt
        tuples = line.split("],")[0]
        tuples = tuples.replace("[", "")
        tuples = list(tuples.replace(" ", "").split(","))
        gt_list += tuples
    return gt_list


def extract_tokens(query):
    tokenizer = RegexpTokenizer(r'\w+')         # Word Tokenizer
    tokens = tokenizer.tokenize(query)
    return tokens


def tokens2embeddings_nomean(tokens, mat, keys):
    embeddings = []
    for token in tokens:
        tt_token = 'tt__'+token
        if tt_token in keys:
            idx = keys.index(tt_token)
            embeddings.append(mat[idx])
            print(f'# Found embedding for token {tt_token} at idx {idx}')
        elif token in keys:
            idx = keys.index(token)
            embeddings.append(mat[idx])
            print(f'# Found embedding for token {token} at idx {idx}')
        else:
            print(f'# No embedding found for token {token}')
    return embeddings


def tokens2embeddings_withmean(tokens, mat, keys):
    embeddings = []
    for token in tokens:
        tt_token = 'tt__'+token
        if token in keys:
            idx = keys.index(token)
            embeddings.append(mat[idx])
            print(f'# Found embedding for token {token} at idx {idx}')
        if tt_token in keys:
            idx = keys.index(tt_token)
            embeddings.append(mat[idx])
            print(f'# Found embedding for token {tt_token} at idx {idx}')
    return embeddings


def compute_similarity_nomean(embeddings, mat):
    sims = []
    for embedding in embeddings:
        emb = np.array(embedding)
        emb = emb.reshape(-1, 1)
        emb_transpose = np.transpose(emb)
        similarity = cosine_similarity(emb_transpose, mat)
        sims.append(similarity)
    return sims


def compute_similarity_withmean(embeddings, mat):
    sims = []
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
        emb = mean_vec
    emb = emb.reshape(1, -1)
    similarity = cosine_similarity(emb, mat)
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


def create_query_embedding(input_file, mat, keys, mean_emb=False):
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
        print(f'# The directory {path} is empty!')
        return -1
    
    df = pd.read_csv(input_file)
    out_df = pd.DataFrame(columns=('File', 'Query', 'Position', 'RID', 'Name', 
                                    'SearchID', 'GT'))

    for file in query_dir:
        if re.match('^\d+', file):
            path_file = path + file
            print(f'# Query extraction from {file}')
            query = extract_query(path_file)
            print(f'# Query extracted: {query}', end='')
            print(f'# Ground_Truth extraction from {file}')
            gt = extract_gt(path_file)
            tokens = extract_tokens(query)
            print(f'# Tokens extracted: {tokens}')
            print('# Embeddings extraction...')
            if mean_emb == False:
                embeddings = tokens2embeddings_nomean(tokens, mat, keys)
            else:
                embeddings = tokens2embeddings_withmean(tokens, mat, keys)
            if len(embeddings) == 0:
                print(f'# No embeddings found for query --> {query}')
            else:
                print('# Computing similarities...')
                if mean_emb == False:
                    similarities = compute_similarity_nomean(embeddings, mat)
                else:
                    similarities = compute_similarity_withmean(embeddings, mat)
                print('# Searching the top5 similar RIDs...')
                top5rids = get_top5RID(similarities, keys)
                for i, idx in enumerate(top5rids):
                    index = int(keys[idx].replace('idx__', ''))
                    name = df.iloc[index-2]['name']
                    searchID = df.iloc[index-2]['__search_id']
                    print(f'{i+1}) --> {keys[idx]} --> {name} --> {searchID}')
                    values_to_add = {'File': file, 'Query': ' '.join(tokens), 
                                    'Position': i+1, 'RID': keys[idx], 
                                    'Name': name, 'SearchID': searchID, 
                                    'GT': 1 if searchID in gt else 0}
                    row_to_add = pd.Series(values_to_add)
                    out_df = out_df.append(row_to_add, ignore_index=True)
                print()
    out_df.to_csv(f'pipeline/debuggings/output_{filename}.csv')
    return 0


if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    # ToDo: Preprocess dataset
    #   dataset_preprocess(input_file, output_file)

    # Create Embedding Matrix
    mat, keys = create_local_embedding(input_file)
    print()

    # Keyword Search
    if create_query_embedding(input_file, mat, keys, mean_emb=False) == -1:
        print('Ops! The function failed!')
