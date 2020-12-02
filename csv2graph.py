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


def create_graph_from_dataset(file):
    g = nx.read_adjlist(file, delimiter=',', create_using=nx.DiGraph())
    nx.draw(g, with_labels=True)
    plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_graph_from_dbms(input_dir):
    pass


if __name__ == '__main__':
    # Check pipeline directory
    assert os.path.isdir('pipeline'), 'Pipeline directory does not exist'
    args = get_arguments()

    if os.path.isdir(args.file):
        # Generate dbms graph
        print('Start creation dbms graph')
        create_graph_from_dbms(args.file)
    else:
        # Read input dataset
        input_file = args.file
        create_graph_from_dataset(input_file)
