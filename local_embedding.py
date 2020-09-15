import os
import datetime
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from EmbDI.embeddings import learn_embeddings
    from EmbDI.sentence_generation_strategies import generate_walks
    from EmbDI.utils import *

    from EmbDI.graph import Graph
    from EmbDI.logging import *
    from edgelist import EdgeList


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


def graph_generation(configuration, edgelist, prefixes, dictionary=None):
    """
    Generate the graph for the given dataframe following the specifications in configuration.
    :return: the generated graph
    """
    # Read external info file to perform replacement.
    if configuration['walks_strategy'] == 'replacement':
        print('# Reading similarity file {}'.format(configuration['similarity_file']))
        list_sim = read_similarities(configuration['similarity_file'])
    else:
        list_sim = None

    if 'flatten' in configuration:
        if configuration['flatten'].lower() not in ['all', 'false']:
            flatten = configuration['flatten'].strip().split(',')
        elif configuration['flatten'].lower() == 'false':
            flatten = []
        else:
            flatten = 'all'
    else:
        flatten = []
    t_start = datetime.datetime.now()
    print(OUTPUT_FORMAT.format('Starting graph construction', t_start.strftime(TIME_FORMAT)))
    if dictionary:
        for __ in edgelist:
            l = []
            for _ in __:
                if _ in dictionary:
                    l.append(dictionary[_])
                else:
                    l.append(_)

        # edgelist_file = [dictionary[_] for __ in edgelist_file for _ in __[:2] if _ in dictionary]
    g = Graph(edgelist=edgelist, prefixes=prefixes, sim_list=list_sim, flatten=flatten)
    t_end = datetime.datetime.now()
    dt = t_end - t_start
    print(OUTPUT_FORMAT.format('Graph construction complete', t_end.strftime(TIME_FORMAT)))
    print(OUTPUT_FORMAT.format('Time required to build graph:', dt.total_seconds()))
    metrics.time_graph = dt.total_seconds()
    return g


def random_walks_generation(configuration, df, graph):
    """
    Traverse the graph using different random walks strategies.
    :param configuration: run parameters to be used during the generation
    :param df: input dataframe
    :param graph: graph generated starting from the input dataframe
    :return: the collection of random walks
    """
    t1 = datetime.datetime.now()
    # Find values in common between the datasets.
    if configuration['intersection']:
        print('# Finding overlapping values. ')
        # Expansion works better when all tokens are considered, rather than only the overlapping ones.
        if configuration['flatten']:
            warnings.warn('Executing intersection while flatten = True.')
        # Find the intersection
        intersection = find_intersection_flatten(df, configuration['dataset_info'])
        if len(intersection) == 0:
            warnings.warn('Datasets have no tokens in common. Falling back to no-intersection.')
            intersection = None
        else:
            print('# Number of common values: {}'.format(len(intersection)))
    else:
        print('# Skipping search of overlapping values. ')
        intersection = None
        # configuration['with_rid'] = WITH_RID_FIRST

    # Generating walks.
    walks = generate_walks(configuration, graph, intersection=intersection)
    t2 = datetime.datetime.now()
    dt = t2 - t1

    metrics.time_walks = dt.total_seconds()
    metrics.generated_walks = len(walks)
    return walks


def embeddings_generation(walks, configuration, dictionary):
    """
    Take the generated walks and train embeddings using the walks as training corpus.
    :param walks:
    :param configuration:
    :param dictionary:
    :return:
    """
    t1 = datetime.datetime.now()
    output_file = configuration['run-tag']

    print(OUTPUT_FORMAT.format('Training embeddings', t1))
    t = 'pipeline/embeddings/' + output_file + '.emb'

    print('File: {}'.format(t))
    learn_embeddings(t, walks, write_walks=configuration['write_walks'],
                     dimensions=int(configuration['n_dimensions']),
                     window_size=int(configuration['window_size']),
                     training_algorithm=configuration['training_algorithm'],
                     learning_method=configuration['learning_method'],
                     sampling_factor=configuration['sampling_factor'])

    if configuration['compression']:
        newf = clean_embeddings_file(t, dictionary)
    else:
        newf = t
    t2 = datetime.datetime.now()
    dt = t2 - t1

    str_ttime = t2.strftime(TIME_FORMAT)
    print(OUTPUT_FORMAT.format('Embeddings generation complete', str_ttime))

    configuration['embeddings_file'] = newf

    metrics.time_embeddings = dt.total_seconds()
    return configuration


def create_local_embedding(input_file):
    output_file = os.path.basename(input_file).split('.')[0]
    configuration = {
        'task': 'train',
        'input_file': input_file,
        'output_file': output_file,

        'flatten': 'tt',
        # 'mlflow': False,
        # 'experiment_type': 'ER',
        # 'match_file': 'pipeline/matches/default/matches-amazon_google.txt',

        'compression': False,
        'ntop': 10,
        'repl_numbers': False,

        'embeddings_file': output_file,
        'n_sentences': 1000,
        'write_walks': True,
    }

    os.makedirs('pipeline/walks', exist_ok=True)
    os.makedirs('pipeline/embeddings', exist_ok=True)

    print('#' * 80)
    print('# Configuration file: {}'.format(configuration['input_file']))
    t_start = datetime.datetime.now()
    print(OUTPUT_FORMAT.format('Starting run.', t_start))

    configuration = check_config_validity(configuration)

    df = pd.read_csv(configuration['input_file'])

    # If the number of tuples is huge, we take the first N/4 tuples
    if (df.shape[0] > 100000):
        numrows = df.shape[0]
        df = df[:int(numrows/12)]

    prefixes = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']

    el = EdgeList(df, prefixes)
    # del el

    # df = pd.read_csv(configuration['input_file'], dtype=str, index_col=False)
    df = el.get_df_edgelist()
    df = df[df.columns[:2]]
    df.dropna(inplace=True)

    run_tag = configuration['output_file']
    configuration['run-tag'] = run_tag

    edgelist = el.get_edgelist()

    # prefixes, edgelist = read_edgelist(configuration['input_file'])

    if configuration['compression']:  # Execute compression if required.
        df, dictionary = dict_compression_edgelist(df, prefixes=prefixes)
        el = df.values.tolist()
    else:
        dictionary = None
        el = edgelist

    graph = graph_generation(configuration, el, prefixes, dictionary)
    # graph = Graph(el.get_edgelist(), prefixes=prefixes)

    if configuration['n_sentences'] == 'default':
        #  Compute the number of sentences according to the rule of thumb.
        configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))

    walks = random_walks_generation(configuration, df, graph)

    del graph  # Graph is not needed anymore, so it is deleted to reduce memory cost
    del el

    configuration = embeddings_generation(walks, configuration, dictionary)

    mat, keys = prepare_emb_matrix(configuration['embeddings_file'])

    t_end = datetime.datetime.now()
    print(OUTPUT_FORMAT.format('Ending run.', t_end))
    dt = t_end - t_start
    print('# Time required: {}'.format(dt.total_seconds()))

    print('')
    print('Embedding Matrix shape: {}'.format(mat.shape))
    print('Keys number: {}'.format(len(keys)))

    print('')
    print('File created:')
    print('1. pipeline/walks/' + configuration['output_file'] + '.walks')
    print('2. pipeline/embeddings/' + configuration['output_file'] + '.emb')

    return mat, keys


if __name__ == '__main__':
    input_file = 'pipeline/datasets/name.csv'
    mat, keys = create_local_embedding(input_file)
