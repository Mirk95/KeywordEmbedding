import re
import datetime
import warnings
import argparse
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from preprocessing.tokenizer import tokenize_dataset, tokenize_sentence
from preprocessing.utils import check_nltk_library, add_nltk_path

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from EmbDI.embeddings import learn_embeddings
    from EmbDI.sentence_generation_strategies import generate_walks
    from EmbDI.utils import *

    from EmbDI.graph import Graph
    from EmbDI.logging import *
    from wrapper.edgelist import EdgeList


def get_embdi_arguments():
    parser = argparse.ArgumentParser(description='Run EmbDI embedding')

    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='dataset or directory to analyze and use to generate token embeddings')

    mode_hint = """
    mode to execute EmbDi algorithm:\n 
    - "full" all step;\n
    - "el" edge list step;\n 
    - "rw" random walk step;\n 
    - "emb" embedding step\n
    """
    parser.add_argument('--mode',
                        type=str,
                        choices=['full', 'el', 'rw', 'emb'],
                        default='full',
                        help=mode_hint)

    args = parser.parse_args()

    if args.mode == 'full' and not os.path.isdir(args.file) and not os.path.isfile(args.file):
        raise ValueError('{} is not a valid file or directory to execute all'.format(args.file))

    if args.mode != 'full' and not os.path.isfile(args.file):
        raise ValueError('{} is not a valid file'.format(args.file))

    return args


def update_edgelist(rename, edgelist, pref='idx__'):
    rename = '{}{}__'.format(pref, rename)
    pattern = '^{}'.format(pref)
    for i, line in enumerate(edgelist):
        n1 = line[0]
        if line[0].startswith(pref):
            n1 = re.sub(pattern, rename, line[0])

        n2 = line[1]
        if line[1].startswith(pref):
            n2 = re.sub(pattern, rename, line[1])

        if n1 != line[0] or n2 != line[1]:
            edgelist[i] = (n1, n2, line[2], line[3])


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


class EmbDIWrapper(object):

    def __init__(self,
                 n_dimensions=300,
                 window_size=3,
                 n_sentences='default',
                 training_algorithm='word2vec',
                 learning_method='skipgram',
                 with_tokenization=True,

                 ignore_columns=None,
                 ):

        # embedding values
        self.mat = np.array([])
        self.keys = []

        # emebedding model parameters
        self.n_dimensions = n_dimensions
        self.window_size = window_size
        self.n_sentences = n_sentences
        self.training_algorithm = training_algorithm
        self.learning_method = learning_method

        # preprocessing data values
        self.with_tokenization = with_tokenization
        self.ignore_columns = ignore_columns if ignore_columns else []

    def fit(self, df, name='test_name'):
        configuration = {
            'task': 'train',
            'input_file': name,
            'output_file': 'embdi__{}'.format(name),

            'flatten': 'tt',

            'compression': False,
            'ntop': 10,
            'repl_numbers': False,

            'embeddings_file': name,
            'n_sentences': self.n_sentences,
            'write_walks': True,

            'n_dimensions': self.n_dimensions,
            'window_size': self.window_size,
            'training_algorithm': self.training_algorithm,
            'learning_method': self.learning_method,
        }

        print('#' * 80)
        t_start = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Starting run.', t_start))

        configuration = check_config_validity(configuration)

        # create a copy of the input dataset
        df = df.copy()

        # ignore selected columns
        if self.ignore_columns is not None:
            for col in self.ignore_columns:
                if col in df:
                    df.drop(col, axis=1, inplace=True)

        # tokenize input dataset
        if self.with_tokenization:
            df = tokenize_dataset(df, stem=True)

        # replace space with underscore
        df = df.applymap(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
        prefixes = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']

        # create edge list
        el = EdgeList(df, prefixes)

        df = el.get_df_edgelist()
        df = df[df.columns[:2]]
        df.dropna(inplace=True)

        run_tag = configuration['output_file']
        configuration['run-tag'] = run_tag

        edgelist = el.get_edgelist()

        # rename edge list idx name
        update_edgelist(name, edgelist, pref='idx__')
        update_edgelist(name, edgelist, pref='cid__')

        # preprocess compression
        if configuration['compression']:  # Execute compression if required.
            df, dictionary = dict_compression_edgelist(df, prefixes=prefixes)
            el = df.values.tolist()

            update_edgelist(name, el, pref='idx__')
            update_edgelist(name, el, pref='cid__')

        else:
            dictionary = None
            el = edgelist

        # graph generation by using edge list
        graph = graph_generation(configuration, el, prefixes, dictionary)

        # define number of sentence
        if configuration['n_sentences'] == 'default':
            #  Compute the number of sentences according to the rule of thumb.
            configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))

        # random walks generation
        walks = random_walks_generation(configuration, df, graph)

        del graph  # Graph is not needed anymore, so it is deleted to reduce memory cost
        del el

        # embedding generation using generated walk
        configuration = embeddings_generation(walks, configuration, dictionary)

        self.mat, self.keys = prepare_emb_matrix(configuration['embeddings_file'])

        t_end = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Ending run.', t_end))
        dt = t_end - t_start
        print('# Time required: {}'.format(dt.total_seconds()))

        print('')
        print('Embedding Matrix shape: {}'.format(self.mat.shape))
        print('Keys number: {}'.format(len(self.keys)))

        print('')
        print('File created:')
        print('1. pipeline/walks/' + configuration['output_file'] + '.walks')
        print('2. pipeline/embeddings/' + configuration['output_file'] + '.emb')

    def fit_db(self, input_dir):
        if not os.path.isdir(input_dir):
            raise ValueError('Input dir does not exists: {}'.format(input_dir))

        # extract dataset name
        dir_name = os.path.normpath(input_dir)
        dir_name = dir_name.split(os.sep)[-1]

        # get list of table name
        file_list = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

        # configuration dictionary
        configuration = {
            'task': 'train',
            'input_file': dir_name,
            'output_file': 'embdi__{}'.format(dir_name),

            'flatten': 'tt',

            'compression': False,
            'ntop': 10,
            'repl_numbers': False,

            'embeddings_file': dir_name,
            'n_sentences': self.n_sentences,
            'write_walks': True,

            'n_dimensions': self.n_dimensions,
            'window_size': self.window_size,
            'training_algorithm': self.training_algorithm,
            'learning_method': self.learning_method,
        }

        print('#' * 80)
        t_start = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Starting run.', t_start))

        configuration = check_config_validity(configuration)

        db_edgelist = []
        prefixes = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']

        run_tag = configuration['output_file']
        configuration['run-tag'] = run_tag

        # for each dataset we compute edgelist file
        for filename in file_list:
            # read dataset
            print('\n filename: {}'.format(filename))

            df = pd.read_csv(os.path.join(input_dir, filename), quotechar='"', error_bad_lines=False)
            # df = df.head(10)

            # ignore selected columns
            if self.ignore_columns is not None:
                for col in self.ignore_columns:
                    if col in df:
                        df.drop(col, axis=1, inplace=True)

            # tokenize dataset
            print('Start tokenization')
            if self.with_tokenization:
                df = tokenize_dataset(df, stem=True)

            # replace space with underscore
            df = df.applymap(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)

            # create edge list
            print('Create edgelist')
            el = EdgeList(df, prefixes)

            # extract edge list
            edgelist = el.get_edgelist()

            # rename edge list idx name
            update_edgelist(filename, edgelist, pref='idx__')
            update_edgelist(filename, edgelist, pref='cid__')

            # add edge list to db_edgelist
            db_edgelist += edgelist

        # create an empty edgelist model

        print('Create global edgelist')
        el = EdgeList(pd.DataFrame(), prefixes)

        # load db_edgelist into el
        el.load_edgelist(db_edgelist)

        # extract dataframe edge list
        df = el.get_df_edgelist()
        df = df[df.columns[:2]]
        df.dropna(inplace=True)

        # extract edge list
        edgelist = el.get_edgelist()

        # preprocess compression
        if configuration['compression']:  # Execute compression if required.
            df, dictionary = dict_compression_edgelist(df, prefixes=prefixes)
            el = df.values.tolist()
        else:
            dictionary = None
            el = edgelist

        # graph generation by using edge list
        graph = graph_generation(configuration, el, prefixes, dictionary)

        # define number of sentence
        if configuration['n_sentences'] == 'default':
            #  Compute the number of sentences according to the rule of thumb.
            configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))

        # random walks generation
        walks = random_walks_generation(configuration, df, graph)

        del graph  # Graph is not needed anymore, so it is deleted to reduce memory cost
        del el

        # embedding generation using generated walk
        configuration = embeddings_generation(walks, configuration, dictionary)

        self.mat, self.keys = prepare_emb_matrix(configuration['embeddings_file'])

        t_end = datetime.datetime.now()
        print(OUTPUT_FORMAT.format('Ending run.', t_end))
        dt = t_end - t_start
        print('# Time required: {}'.format(dt.total_seconds()))

        print('')
        print('Embedding Matrix shape: {}'.format(self.mat.shape))
        print('Keys number: {}'.format(len(self.keys)))

        print('')
        print('File created:')
        print('1. pipeline/walks/' + configuration['output_file'] + '.walks')
        print('2. pipeline/embeddings/' + configuration['output_file'] + '.emb')

    def load_embedding(self, embedding_file):
        self.mat, self.keys = prepare_emb_matrix(embedding_file)

    def preprocess_sentence(self, sentence):
        if self.with_tokenization:
            sentence = tokenize_sentence(sentence, stem=True)

        return sentence

    def get_token_embedding(self, token):
        vec = None
        if 'tn__{}'.format(token) in self.keys:
            token = 'tn__{}'.format(token)
            vec = self.mat[self.keys.index(token)]

        elif 'tt__{}'.format(token) in self.keys:
            token = 'tt__{}'.format(token)
            vec = self.mat[self.keys.index(token)]

        return vec

    def get_sentence_embedding(self, sentence, keep_none=False):
        vecs = []
        sentence = sentence.split(' ')
        for word in sentence:
            vec = self.get_token_embedding(word)

            if vec is not None or keep_none:
                vecs.append(vec)

        return vecs

    def get_k_nearest_token(self, sentence, k=5, distance='cosine', pref='idx', withMean=True):
        cond = [True if x.startswith(pref) else False for x in self.keys]

        emb_sentence = self.get_sentence_embedding(sentence)
        if not emb_sentence:
            return []
        emb_sentence = np.array(emb_sentence)

        if withMean:
            emb_sentence = np.mean(emb_sentence, axis=0, keepdims=True)

        if distance == 'cosine':
            distance_matrix = cosine_distances(emb_sentence, self.mat[cond])
        elif distance == 'euclidean':
            distance_matrix = euclidean_distances(emb_sentence, self.mat[cond])
        else:
            raise ValueError('Selected the wrong distance {}'.format(distance))

        if not withMean:
            distance_matrix = np.sum(distance_matrix, axis=0, keepdims=True)

        distance_matrix = distance_matrix.ravel()

        indexes = distance_matrix.argsort()[:k]
        keys = np.array([self.keys[i] for i in range(len(self.keys)) if cond[i]])
        new_keys = keys[indexes]

        return new_keys


def test_record_id_embedding(wrapper, df):
    print('#' * 60)
    print('Start Testing')

    # emb = get_token_embedding('Robert', mat, keys)

    # Select records for testing
    max_record = 5
    selected_record = []
    for idx, val in df.iterrows():
        res = val['name'].split(' ')
        if len(res) >= 2 and len(res[0]) > 4 and len(res[1]) > 4:
            selected_record.append(idx)

        if len(selected_record) >= max_record:
            break

    print('Selected {} records'.format(len(selected_record)))

    # Evaluate neighbour for each record
    for idx in selected_record:
        sentence = df.loc[idx, 'name']
        print('\nidx {} sentence: {}'.format(idx, sentence))

        neighbours = wrapper.get_k_nearest_token(sentence, k=5, distance='cosine', pref='idx', withMean=False)
        neighbours = [int(x.replace('idx__', '')) for x in neighbours]
        print(df.loc[neighbours, 'name'])

    print('End :(')


if __name__ == '__main__':
    # Add nltk data directory
    add_nltk_path('/Users/francesco/Development')
    add_nltk_path('pipeline/nlp/')

    # check nltk library dependency
    check_nltk_library()

    # Check pipeline directory
    assert os.path.isdir('pipeline'), 'Pipeline directory does not exist'

    # Create mandatory dir for EmbDi
    os.makedirs('pipeline/edges', exist_ok=True)
    os.makedirs('pipeline/walks', exist_ok=True)
    os.makedirs('pipeline/embeddings', exist_ok=True)

    # conf paramss
    with_tokenization = True
    ignore_columns = ['__search_id']
    blacklist_columns = ['']
    prefixes = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']

    args = get_embdi_arguments()

    if args.mode == 'full':
        # define model
        wrapper = EmbDIWrapper(ignore_columns=ignore_columns)

        if os.path.isdir(args.file):
            # generate dbms embedding
            print('Start embedding dbms')
            wrapper.fit_db(args.file)

        else:
            # Read input dataset
            input_file = args.file
            print('Read input dataset')
            df = pd.read_csv(input_file)

            file_name = str(os.path.basename(input_file).split('.')[0])

            # tokenize dataset
            print('Tokenize input dataset')
            new_df = tokenize_dataset(df, stem=True)

            # generate embedding
            print('Start embedding dataset')
            wrapper.fit(new_df, name=file_name)

    elif args.mode == 'el':
        # Read input dataset
        input_file = args.file
        print('Read input dataset')
        df = pd.read_csv(input_file)

        filename = str(os.path.basename(input_file).split('.')[0])

        # Ignore selected columns
        for col in ignore_columns:
            if col in df:
                df.drop(col, axis=1, inplace=True)

        # Tokenize input dataset
        if with_tokenization:
            print('Tokenize input dataset')
            df = tokenize_dataset(df, stem=True)

        # Generate edge list
        print('Generate Edge List')
        el = EdgeList(df, prefixes)
        el = el.get_edgelist()

        # Loading the graph to make sure it can load the edgelist.
        # print('Check Edge List')
        # g = Graph(el, prefixes=prefixes)

        # Update edgelist idx
        print('Update Edge List')
        update_edgelist(filename, el, pref='idx__')
        update_edgelist(filename, el, pref='cid__')

        # Save Edge List
        print('Save Edge List')
        filename = filename + '.edges'
        filename = os.path.join('pipeline/edges', filename)

        pd.DataFrame(el, dtype=str, columns=prefixes).to_csv(filename, index=False)

    elif args.mode == 'rw':

        # Read edge list
        print('Read Edge List')
        df = pd.read_csv(args.file)
        prefixes, edgelist = read_edgelist(args.file)

        configuration = {
            'walks_strategy': 'basic',
            'flatten': 'tt',
            'n_sentences': 'default',
            'sentence_length': 10,
            'write_walks': True,
            'intersection': False,
            'backtrack': True,
            'output_file': str(os.path.basename(args.file).split('.')[0]),
            'repl_numbers': False,
            'repl_strings': False,
            'factor': 100,
        }

        # Create graph from edge list
        print('Create Graph')
        graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)

        if configuration['n_sentences'] == 'default':
            #  Compute the number of sentences according to the rule of thumb.
            configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']),
                                                                     factor=configuration['factor'])

        # Generate random walk and save it
        print('Generate Random Walk')
        walks = random_walks_generation(configuration, df, graph)

        print('Finish: {}'.format(walks))

    elif args.mode == 'emb':

        configuration = {
            'run-tag': 'embdi__' + str(os.path.basename(args.file).split('.')[0]),
            'write_walks': True,
            'n_dimensions': 300,
            'window_size': 3,
            'training_algorithm': 'word2vec',
            'learning_method': 'skipgram',
            'sampling_factor': 0.001,
            'compression': False,
        }

        # Embedding generation using generated walk
        print('Generate Embedding')
        configuration = embeddings_generation(args.file, configuration, None)

    print(':)')
