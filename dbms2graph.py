import os
import json
import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from preprocessing.tokenizer import tokenize_dataset


def get_arguments():
    parser = argparse.ArgumentParser(description='Run csv2graph script')
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='dataset or directory to analyze and use to generate the graph')
    args = parser.parse_args()
    return args


def get_schema(schema_dir):
    """
    :param schema_dir: path to schema directory
    :return schema: a dictionary with schema
    """
    db_schema_path = schema_dir + 'db_schema.json'
    if os.path.isfile(db_schema_path):
        with open(db_schema_path, 'r') as f:
            db_json_data = json.load(f)
    else:
        raise ValueError(f'ERROR: Not found the schema file inside the directory {schema_dir}!')

    schema = dict()
    for key in db_json_data.keys():
        # Remove duplicated dictionaries inside list
        schema[key] = [dict(t) for t in {tuple(d.items()) for d in db_json_data[key]}]

    return schema


def nodes_and_edges_from_df_extraction(table_name, dataframe, G, labels):
    for idx, row in dataframe.iterrows():
        node_name = table_name + '__' + str(idx)
        G.add_node(node_name)
        row_dict = row.to_dict()
        values = [str(table_name) + '.' + str(key) + '#' + str(row_dict[key])[:20]
                  for key in row_dict.keys() if row_dict[key] != '']
        columns = ['val->' + col for col in row_dict.keys() if row_dict[col] != '']
        G.add_nodes_from(values)
        combinations_edges = [(node_name, i) for i in values]
        G.add_edges_from(combinations_edges)
        labels_row = dict(zip(combinations_edges, columns))
        labels.update(labels_row)


def foreign_keys_extraction(input_dir, table_name, table_schema, G, labels, with_tokenization):
    for relation in table_schema:
        if relation['constraint_type'] == 'FOREIGN KEY':
            column_name = relation['column_name']
            foreign_table_name = relation['foreign_table_name']
            foreign_column_name = relation['foreign_column_name']
            df1 = pd.read_csv(input_dir + table_name + '.csv', na_filter=False)
            df1 = df1[:50]
            df2 = pd.read_csv(input_dir + foreign_table_name + '.csv', na_filter=False)
            df2 = df2[:50]
            if with_tokenization:
                df1 = tokenize_dataset(df1, stem=True)
                df2 = tokenize_dataset(df2, stem=True)
            df1 = df1.applymap(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
            df2 = df2.applymap(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
            df1 = df1.astype(str)
            df2 = df2.astype(str)
            column_names1 = df1.columns.tolist()
            column_names2 = df2.columns.tolist()
            list1_as_set = set(column_names1)
            intersection = list1_as_set.intersection(column_names2)
            intersection_as_list = list(intersection)
            result = pd.merge(df1, df2, left_on=column_name, right_on=foreign_column_name)
            if (column_name in intersection_as_list) and (foreign_column_name in intersection_as_list):
                result = result[[column_name + '_x', foreign_column_name + '_y']]
            elif (column_name not in intersection_as_list) and (foreign_column_name not in intersection_as_list):
                result = result[[column_name, foreign_column_name]]
            elif (column_name in intersection_as_list) and (foreign_column_name not in intersection_as_list):
                result = result[[column_name + '_x', foreign_column_name]]
            else:
                result = result[[column_name, foreign_column_name + '_y']]
            if not result.empty:
                result = result.drop_duplicates()
                for _, row in result.iterrows():
                    first_node = table_name + '.' + column_name + '#' + str(row[0])
                    second_node = foreign_table_name + '.' + foreign_column_name + '#' + str(row[1])
                    if G.has_edge(first_node, second_node) or G.has_edge(second_node, first_node):
                        pass
                    else:
                        G.add_edge(first_node, second_node)
                        new_label = {(first_node, second_node): 'FK'}
                        labels.update(new_label)


def create_graph(input_dir, with_tokenization=False):
    if not os.path.isdir(input_dir):
        raise ValueError('Input dir does not exists: {}'.format(input_dir))

    # Get list of table name
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    schema = get_schema('pipeline/schemas/')
    G = nx.Graph()
    labels_value = dict()
    labels_fk = dict()
    for filename in file_list:
        # Read dataset
        print('Filename: {}'.format(filename))
        df = pd.read_csv(os.path.join(input_dir, filename), na_filter=False)
        df = df[:50]
        # Tokenize dataset
        print('Start tokenization')
        if with_tokenization:
            df = tokenize_dataset(df, stem=True)
        # Replace space with underscore
        df = df.applymap(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
        table_name = filename.split('.')[0]
        print('Start nodes and edges extraction from dataframe...')
        nodes_and_edges_from_df_extraction(table_name, df, G, labels_value)

    for table in schema.keys():
        print('Start Foreign Keys extraction for {} table'.format(table))
        foreign_keys_extraction(input_dir, table, schema[table], G, labels_fk, with_tokenization)

    labels_value.update(labels_fk)
    pos = nx.spring_layout(G)
    deg = dict(G.degree)
    nx.draw(G, pos, node_size=[v * 5 for v in deg.values()], with_labels=False)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_fk, font_size=8)
    plt.savefig('graph.png', dpi=300, bbox_inches='tight')
    plt.show()
    return G


if __name__ == '__main__':
    # Check pipeline directory
    assert os.path.isdir('pipeline'), 'Pipeline directory does not exist'
    args = get_arguments()
    # Generate dbms graph
    print('Start creation dbms graph')
    graph = create_graph(args.file, with_tokenization=True)
