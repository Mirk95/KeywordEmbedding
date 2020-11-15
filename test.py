import numpy as np

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


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


mat, keys = prepare_emb_matrix('pipeline/output/retrofitted_vectors.wv')
idx_element = keys.index("name.name#Koch_Eckehard")
emb = mat[idx_element]
emb = emb.reshape(1, -1)
distance_matrix_cosine = cosine_distances(emb, mat)
distance_matrix_euclidean = euclidean_distances(emb, mat)

distance_matrix_cosine = np.sum(distance_matrix_cosine, axis=0, keepdims=True)
distance_matrix_cosine = distance_matrix_cosine.ravel()

distance_matrix_euclidean = np.sum(distance_matrix_euclidean, axis=0, keepdims=True)
distance_matrix_euclidean = distance_matrix_euclidean.ravel()

indexes_cosine = distance_matrix_cosine.argsort()[:20]
indexes_euclidean = distance_matrix_euclidean.argsort()[:20]

k = np.array(keys)
new_keys_cosine = k[indexes_cosine]
new_keys_euclidean = k[indexes_euclidean]

print('COSINE: ', new_keys_cosine)
print('EUCLIDEAN: ', new_keys_euclidean)