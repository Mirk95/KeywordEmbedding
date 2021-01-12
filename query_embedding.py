import os
import re
import nltk
import argparse
import warnings
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


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
    parser.add_argument('--mode',
                        type=str,
                        choices=['single-to-single', 'single-to-many'],
                        default='single-to-single',
                        help='Mode to use for query embedding')
    parser.add_argument('--clustering_method',
                        type=str,
                        choices=['elbow', 'silhouette', 'offline'],
                        default='offline',
                        help='Clustering method to use for query embedding')
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


# @Francesco's Function
def extract_representatives(X, cluster_index, cluster_size_dict, representatives):
    num_clusters = len(set(cluster_index))
    for clu in range(num_clusters):
        clu_idx = np.argwhere(cluster_index == clu + 1)[:, 0]
        cluster_size_dict[clu] = clu_idx.shape[0]
        representative_center = np.average(X[clu_idx, :], axis=0)
        representatives.append(representative_center)


# @Francesco's Function
def offline_clustering(X, cluster_size_dict, representatives, max_dist=0.3):
    print('Starting offline clustering...')
    p_dist = pdist(X)
    z = linkage(p_dist, 'complete')
    cluster_index = fcluster(z, max_dist, criterion='distance')
    extract_representatives(X, cluster_index, cluster_size_dict, representatives)
    print('Processed {} instances.'.format(X.shape[0]))
    print('Found {} clusters offline.\n'.format(len(representatives)))
    return cluster_index


# Function returns WSS score for k values from 1 to k_max
def calculate_WSS(data, k_max):
    sse = []
    for k in range(1, k_max+1):
        k_means = KMeans(n_clusters=k).fit(data)
        centroids = k_means.cluster_centers_
        predicted_clusters = k_means.predict(data)
        curr_sse = 0

        # Calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(data)):
            curr_center = centroids[predicted_clusters[i]]
            curr_sse += (data[i, 0] - curr_center[0]) ** 2 + (data[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


def calculate_silhouette(data, k_max):
    sil = []
    # Dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, k_max+1):
        k_means = KMeans(n_clusters=k).fit(data)
        labels = k_means.labels_
        sil.append(silhouette_score(data, labels, metric='euclidean'))
    return sil


def plot_and_save_graph(min_range, max_range, data, x_label, y_label, title, filename):
    style.use("fivethirtyeight")
    plt.plot(range(min_range, max_range), data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    figure = plt.gcf()
    figure.set_size_inches(14, 10)
    plt.savefig(filename + '.png', dpi=100)
    plt.show()


def find_optimal_clusters(matrix, method):
    k_max = 10
    if method == 'elbow':
        """
            Calculate the Within-Cluster-Sum of Squared Errors (WSS) for different values of k, 
            and choose the k for which WSS becomes first starts to diminish. 
            In the plot of WSS-versus-k, this is visible as an elbow.
            Where:
            -   The Squared Error for each point is the square of the distance of the point from its representation 
                i.e. its predicted cluster center.
            -   The WSS score is the sum of these Squared Errors for all the points.
            -   Any distance metric like the Euclidean Distance or the Manhattan Distance can be used.
        """
        sse = calculate_WSS(matrix, k_max)
        differences = [sse[i] - sse[i+1] for i in range(len(sse) - 1)]
        differences = [1 if i < 0 else i for i in differences]
        best_k = differences.index(min(differences)) + 1
        plot_and_save_graph(1, (k_max+1), sse,
                            'Number of Clusters (k)', 'WSS',
                            'Elbow method for optimal k', 'elbow_graph')
        print(f"Optimal K with Elbow Method = {best_k}")
        return best_k
    elif method == 'silhouette':
        """
            The silhouette value measures how similar a point is to its own cluster (cohesion) 
            compared to other clusters (separation). The range of the Silhouette value is between +1 and -1. 
            A high value is desirable and indicates that the point is placed in the correct cluster. 
            If many points have a negative Silhouette value, it may indicate that we have created too many 
            or too few clusters.
        """
        silhouette = calculate_silhouette(matrix, k_max)
        best_k = silhouette.index(max(silhouette)) + 2
        plot_and_save_graph(2, (k_max+1), silhouette,
                            'Number of Clusters (k)', 'Silhouette Score',
                            'Silhouette method for optimal k', 'silhouette_graph')
        print(f"Optimal K with Silhouette Method = {best_k}")
        return best_k
    else:
        raise ValueError("Error: type of method to find optimal clusters unknown!")


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

        # Plot embeddings:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(wrapper.mat)
        t = reduced.transpose()
        plt.scatter(t[0], t[1])
        figure = plt.gcf()
        figure.set_size_inches(14, 10)
        plt.savefig('embeddings.png', dpi=100)
        plt.show()

        if args.mode == 'single-to-many':
            # Compute clusters only once, instead computing it for each query
            if args.clustering_method == 'offline':
                cluster_size_dict = dict()
                representatives = list()
                cluster_index = offline_clustering(wrapper.mat[:100], cluster_size_dict, representatives)
                k = len(representatives)
            else:
                k = find_optimal_clusters(wrapper.mat[:100], method=args.clustering_method)
                kmeans = KMeans(n_clusters=k).fit(wrapper.mat[:100])
                centroids = kmeans.cluster_centers_
                cluster_index = kmeans.labels_

        for label_name in label_files:
            print('\n File: {}'.format(label_name))
            keywords, search_ids = extract_keywords(os.path.join(label_dir, label_name))
            print('Keywords {}'.format(keywords))
            print('Label search id {}'.format(search_ids))
            print('\n Embedding extraction')

            # Preprocess sentence with trained model preprocessing
            sentence = wrapper.preprocess_sentence(keywords)

            if args.mode == 'single-to-single':
                # Get nearest_neighbour
                print('\n Get K nearest records')
                neighbours = wrapper.get_k_nearest_token(sentence, k=5, distance='cosine', pref='idx', withMean=False)
                neighbours = [x.replace('idx__', '') for x in neighbours]
                for neighbour in neighbours:
                    table, idx = neighbour.split('__')
                    df = pd.read_csv('pipeline/datasets/' + table + '.csv', na_filter=False)
                    print(df.loc[int(idx)])
            else:
                # Single-to-many mode
                pass

    else:
        raise ValueError(f'The file {emb_file} does not exist!')
