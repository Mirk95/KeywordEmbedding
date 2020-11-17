import os
import re
import numpy as np
import pandas as pd
from os import linesep


def prepare_emb_matrix(embeddings_file):
    # Reading the reduced file
    keys = []
    with open(embeddings_file, 'r') as fp:
        lines = fp.readlines()

        sizes = lines[0].split()
        sizes = [int(_) for _ in sizes]

        mat = np.zeros(shape=sizes)
        for n, line in enumerate(lines[1:]):
            ll = line.strip().split()
            mat[n, :] = np.array(ll[1:])
            keys.append(ll[0])
    return mat, keys


def tokenize(term):
    if type(term) == str:
        return re.sub('[\.#~\s,\(\)/\[\]:]+', '_', str(term))
    else:
        return ''


def output_vectors(term_list, matrix, output_file_name):
    # Init output file
    f_out = open(output_file_name, 'w')
    # Write meta information
    f_out.write('%d %d' % (matrix.shape[0], matrix.shape[1]) + linesep)
    # Write term vector pairs
    for i, term in enumerate(term_list):
        if i % 1000 == 0:
            print('Exported', i, 'term vectors | Current term:', term)
        f_out.write('%s %s' % (term, ' '.join([str(x) for x in matrix[i]])))
        f_out.write(linesep)


mat, keys = prepare_emb_matrix('pipeline/output/retrofitted_vectors.wv')
datasets_directory = 'pipeline/datasets/'
blacklist = ['char_name', 'movie_info', 'role_type', 'title']
counter = 0

for filename in os.listdir(datasets_directory):
    name = filename.split('.')[0]
    if name not in blacklist:
        df = pd.read_csv(datasets_directory + filename, na_filter=False)
        columns = df.columns.tolist()
        for idx, row in df.iterrows():
            embeddings = []
            for col in columns:
                key = name + '.' + col + '#' + tokenize(str(row[col]))
                if key in keys:
                    # Found Embedding
                    index_key = keys.index(key)
                    embeddings.append(mat[index_key])
            if len(embeddings) > 1:
                new_embedding_idx = name + '.idx#' + str(idx)
                new_embedding = np.mean(embeddings, axis=0, keepdims=True)
                keys.append(new_embedding_idx)
                mat = np.append(mat, new_embedding, axis=0)
                counter += 1

output_vectors(keys, mat, 'pipeline/output/retrofitted_vectors_with_idx.wv')
