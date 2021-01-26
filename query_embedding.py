import os
import re
import nltk
import pickle
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
    from preprocessing.tokenizer import tokenize_dataset
    from dbms2graph import create_graph


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
    parser.add_argument('--approach',
                        type=str,
                        choices=['cn_search', 'clustering'],
                        default='cn_search',
                        help='Approach to use for Single-to-Many mode')
    parser.add_argument('--cn_technique',
                        type=str,
                        choices=['base', 'first'],
                        default='base',
                        help='Technique used for Candidate Network Search')
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
def offline_clustering(X, max_dist=0.3):
    print('Starting offline clustering...')
    p_dist = pdist(X)
    z = linkage(p_dist, 'complete')
    clusters_indices = fcluster(z, max_dist, criterion='distance')
    num_clusters = len(set(clusters_indices))
    print('Processed {} instances.'.format(X.shape[0]))
    print('Found {} clusters offline.\n'.format(num_clusters))
    return num_clusters, clusters_indices


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
    k_max = 100
    print("Start finding optimal k value...")
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


def plot_embeddings(matrix):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(matrix)
    t = reduced.transpose()
    plt.scatter(t[0], t[1])
    figure = plt.gcf()
    figure.set_size_inches(14, 10)
    plt.savefig('embeddings.png', dpi=100)
    plt.show()


def candidate_network_search(datasets_dir, record, graph, sentence, technique):
    search_ids_found = []
    table, idx = record.split('__')
    df = pd.read_csv(datasets_dir + table + '.csv', na_filter=False)
    search_ids_found.append(df.loc[int(idx), '__search_id'])
    print(df.loc[int(idx)])
    print('\n')

    # Get list of tables names
    file_list = [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]
    num_step = len(file_list)

    candidate_network = [record]
    rec = record

    if technique == 'base':
        # BASE APPROACH
        for _ in range(num_step):
            list_records_linked = list(graph[rec])      # Find linked records
            if not list_records_linked:
                break

            for i in range(len(list_records_linked)):
                new_record = list_records_linked[i]
                if new_record not in candidate_network:
                    candidate_network.append(new_record)
                    table_edge, idx_edge = new_record.split('__')
                    print(f"Found an edge --> table: {table_edge} and idx: {idx_edge}")
                    df = pd.read_csv(datasets_dir + table_edge + '.csv', na_filter=False)
                    search_ids_found.append(df.loc[int(idx_edge), '__search_id'])
                    print(df.loc[int(idx_edge)])
                    print('\n')
                    rec = new_record
                    break
        return search_ids_found
    else:
        # FIRST APPROACH
        for _ in range(num_step):
            list_records_linked = list(graph[rec])          # Find linked records
            if not list_records_linked:
                break

            list_records = ['idx__' + r for r in list_records_linked]
            new_record = wrapper.get_best_record(sentence, list_records, k=1, distance='cosine', withMean=True)
            new_record = str(new_record[0])
            new_record = new_record.replace('idx__', '')
            if new_record not in candidate_network:
                candidate_network.append(new_record)            # Append this record to CN list
                table_edge, idx_edge = new_record.split('__')
                print(f"Found an edge --> table: {table_edge} and idx: {idx_edge}")
                df = pd.read_csv(datasets_dir + table_edge + '.csv', na_filter=False)
                search_ids_found.append(df.loc[int(idx_edge), '__search_id'])
                print(df.loc[int(idx_edge)])
                print('\n')
            rec = new_record
        return search_ids_found


def query_embeddings(arguments, embeddings_file):
    results = {}
    # Load embedding matrix
    wrapper.load_embedding(embeddings_file)
    # Plot embeddings
    # plot_embeddings(wrapper.mat)

    label_dir = 'pipeline/queries/IMDB'
    datasets_dir = 'pipeline/datasets/'
    label_files = ["{:03d}.txt".format(x) for x in range(1, 51)]

    if arguments.mode == 'single-to-many':
        # Compute the graph of dbms, useful for both cn_search and clustering approaches
        print("Starting creation of dbms graph...")
        graph = create_graph(datasets_dir, with_tokenization=True)
        print("Dbms graph successfully created!")
        print('\n')

        if arguments.approach == 'clustering':
            # Compute clusters only once, instead computing it for each query
            print("Starting clustering...")
            cond = [True if x.startswith('idx') else False for x in wrapper.keys]
            if arguments.clustering_method == 'offline':
                k, clusters_indices = offline_clustering(wrapper.mat[cond])
            else:
                k = find_optimal_clusters(wrapper.mat[cond], method=arguments.clustering_method)
                kmeans = KMeans(n_clusters=k).fit(wrapper.mat[cond])
                clusters_indices = kmeans.labels_

    for label_name in label_files:
        print('#' * 80)
        print('File: {}'.format(label_name))
        keywords, search_ids = extract_keywords(os.path.join(label_dir, label_name))
        print('Keywords: {}'.format(keywords))
        print('Label search id: {}'.format(search_ids))
        print('Embedding extraction...')

        # Preprocess sentence with trained model preprocessing
        sentence = wrapper.preprocess_sentence(keywords)

        print('Get K nearest records:')
        search_ids_found = []
        neighbours = wrapper.get_k_nearest_token(sentence, k=5, distance='cosine', pref='idx', withMean=True)
        neighbours = [x.replace('idx__', '') for x in neighbours]

        if arguments.mode == 'single-to-single':
            # Single-to-Single mode
            for neighbour in neighbours:
                table, idx = neighbour.split('__')
                if '.csv' in table:
                    table = table.replace('.csv', '')
                df = pd.read_csv(datasets_dir + table + '.csv', na_filter=False)
                print(df.loc[int(idx)])
                print('\n')
                search_ids_found.append(df.loc[int(idx), '__search_id'])
            results[label_name] = (keywords, search_ids, search_ids_found)
        else:
            # Single-to-Many mode
            for neighbour in neighbours:
                table, idx = neighbour.split('__')
                if '.csv' in table:
                    table = table.replace('.csv', '')
                if arguments.approach == 'cn_search':
                    # Candidate Network Searching approach
                    node_name = table + '__' + idx
                    print(f"Searching Candidate Network for node {node_name}: ")
                    search_ids_neighbour = candidate_network_search(datasets_dir, node_name, graph,
                                                                    sentence, technique=arguments.cn_technique)
                    search_ids_found.append(search_ids_neighbour)
                else:
                    # Clustering approach
                    df = pd.read_csv(datasets_dir + table + '.csv', na_filter=False)
                    df = tokenize_dataset(df, stem=False)
                    print(df.loc[int(idx)])
                    print('\n')
                    search_ids_found.append(df.loc[int(idx), '__search_id'])
                    keys_with_only_idx = [wrapper.keys[i] for i in range(len(wrapper.keys)) if cond[i]]
                    key_index = keys_with_only_idx.index('idx__' + table + '__' + str(idx))
                    cluster = clusters_indices[key_index]
                    items_list = list(np.array(np.where(clusters_indices == cluster)).ravel())
                    for item in items_list:
                        row = keys_with_only_idx[item]
                        row = row.replace('idx__', '')
                        table_item, idx_item = row.split('__')
                        df = pd.read_csv(datasets_dir + table_item + '.csv', na_filter=False)
                        print(df.loc[int(idx_item)])
                        print('\n')
                        search_ids_found.append(df.loc[int(idx_item), '__search_id'])
            results[label_name] = (keywords, search_ids, search_ids_found)
    return results


if __name__ == '__main__':
    # Add nltk data directory
    nltk.data.path.append('/home/mirko/nltk_data')

    # Retrieve Arguments
    args = get_arguments()

    # Define embedding model
    if args.wrapper == 'embdi':
        wrapper = EmbDIWrapper(n_dimensions=300, window_size=3, n_sentences='default',
                               training_algorithm='word2vec', learning_method='skipgram',
                               with_tokenization=True)
    elif args.wrapper == 'retro':
        wrapper = RETRONumericWrapper(n_iterations=10, alpha=1.0, beta=0.0, gamma=3.0, delta=1.0,
                                      number_dims=300, standard_deviation=1.0, table_blacklist=[],
                                      with_tokenization=True)
    else:
        wrapper = BaseWrapper(training_algorithm='word2vec_CBOW', n_dimensions=300, window_size=3,
                              with_tokenization=True, ignore_columns=None, insert_col=True,
                              permutation_rate=10)

    if args.wrapper == 'base':
        emb_file = 'pipeline/embeddings/' + args.wrapper + '_' + wrapper.training_algorithm + '__datasets.emb'
    else:
        emb_file = 'pipeline/embeddings/' + args.wrapper + '__datasets.emb'

    if os.path.isfile(emb_file):
        results_dir = 'pipeline/query_emb_results/'
        os.makedirs(results_dir, exist_ok=True)

        final_results = query_embeddings(args, emb_file)
        # Saving results in file pickle
        if args.wrapper == 'base':
            name_wrapper = args.wrapper + str(wrapper.training_algorithm).replace('_', '')
        else:
            name_wrapper = args.wrapper

        if args.mode == 'single-to-single':
            query_emb_filename = name_wrapper + '_singletosingle.pickle'
        else:
            query_emb_filename = name_wrapper + '_cnsearch_' + args.cn_technique + '.pickle'

        with open(results_dir + query_emb_filename, 'wb') as handle:
            pickle.dump(final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f'The file {emb_file} does not exist!')
