import os
import argparse
import networkx as nx
import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser(description='Run csv2graph script')
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='dataset or directory to analyze and use to generate the graph')

    args = parser.parse_args()
    return args


def create_graph(input_dir):
    if not os.path.isdir(input_dir):
        raise ValueError('Input dir does not exists: {}'.format(input_dir))

    # Extract dataset name
    dir_name = os.path.normpath(input_dir)
    dir_name = dir_name.split(os.sep)[-1]

    # Get list of table name
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    file_names = [f.split('.')[0] for f in file_list]
    G = nx.Graph()
    G.add_nodes_from(file_names)
    return G


if __name__ == '__main__':
    # Check pipeline directory
    assert os.path.isdir('pipeline'), 'Pipeline directory does not exist'
    args = get_arguments()
    # Generate dbms graph
    print('Start creation dbms graph')
    graph = create_graph(args.file)
