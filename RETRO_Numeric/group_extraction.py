#!/usr/bin/python3

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

import RETRO_Numeric.retro_utils as utils
import RETRO_Numeric.encoding_utils as encoder


def get_graph(path, graph_type='gml'):
    g = None
    if graph_type == 'gml':
        g = nx.read_gml(path)
    return g


def get_group(name, group_type, vector_dict, extended=None, query='', export_type='full', data_type='string'):
    elements = []
    if group_type == 'categorial':
        if export_type == 'full':
            elements = vector_dict
        else:
            elements = list(vector_dict.keys())
    else:
        elements = vector_dict  # Vector_dict is just the count of the elements
    result = {
        'name': name,
        'type': group_type,
        'elements': elements,
        'query': query,
        'data_type': data_type  # 'string' or 'number'
    }
    if extended is not None:
        if export_type == 'full':
            result['inferred_elements'] = extended
        else:
            result['inferred_elements'] = list(extended.keys())
    return result


def get_column_groups(df_vectors, graph, terms, conf):
    print("Column relation extraction started:")
    result = dict()
    tokenization_settings = conf['TOKENIZATION_SETTINGS']
    # Initialize tokenization algorithms
    initialize_numeric_tokenization(df_vectors, tokenization_settings)

    for node in graph.nodes:
        columns_attr = graph.nodes[node]['columns']
        types_attr = graph.nodes[node]['types']
        column_names = zip(columns_attr, types_attr) if type(columns_attr) == list and type(types_attr) == list \
            else [(columns_attr, types_attr)]
        for column_name, column_type in column_names:
            print('Processing %s.%s ...' % (node, column_name))
            vec_dict_fit = dict()
            vec_dict_inferred = dict()

            # Process numeric values
            if column_type == 'number':
                get_numeric_column_groups(node, column_name, vec_dict_inferred, terms, tokenization_settings, conf)
            else:  # Process string values
                df_node = pd.read_csv(conf['DATASETS_PATH'] + str(node) + '.csv', na_filter=False)
                df_node = df_node.applymap(str)
                df_node = df_node.applymap(lambda x: utils.tokenize(x) if isinstance(x, str) else x)
                merging = pd.merge(df_node, df_vectors, how='left', left_on=column_name, right_on=df_vectors.word)
                merging = merging[[column_name, 'vector', 'id_vec']]
                merging = merging.fillna('')
                records = merging.to_records(index=False)
                term_vecs = list(records)
                for (term, vec_bytes, vec_id) in term_vecs:
                    # Modified the following rows to support multi-words per token
                    for val in term.split('_'):
                        inferred, vector = encoder.text_to_vec(val, vec_bytes, terms, tokenization_settings)
                        if inferred:
                            if vector is None:
                                continue
                            vec_dict_inferred[val] = dict()
                            vec_dict_inferred[val]['vector'] = vector
                        else:
                            vec_dict_fit[val] = dict()
                            vec_dict_fit[val]['vector'] = np.array(vector.split(), dtype='float32')
                            vec_dict_fit[val]['id'] = int(vec_id)
                    # inferred, vector = encoder.text_to_vec(term, vec_bytes, terms, tokenization_settings)
                    # if inferred:
                    #     if vector is None:
                    #         continue
                    #     vec_dict_inferred[term] = dict()
                    #     vec_dict_inferred[term]['vector'] = vector
                    # else:
                    #     vec_dict_fit[term] = dict()
                    #     vec_dict_fit[term]['vector'] = np.array(vector.split(), dtype='float32')
                    #     vec_dict_fit[term]['id'] = int(vec_id)

            result['%s.%s' % (node, column_name)] = [get_group('%s.%s' % (node, column_name),
                                                               'categorial',
                                                               vec_dict_fit,
                                                               extended=vec_dict_inferred,
                                                               data_type=column_type)]
    return result


def initialize_numeric_tokenization(df_vectors, tokenization_strategy):
    if tokenization_strategy == 'we-regression':
        encoder.initialize_numeric_word_embeddings(df_vectors)


def get_numeric_column_groups(node, column_name, vec_dict, terms, tokenization_settings, conf):
    mode = tokenization_settings["NUMERIC_TOKENIZATION"]["MODE"]
    buckets = encoder.set_buckets(tokenization_settings["NUMERIC_TOKENIZATION"]["BUCKETS"])
    normalization = encoder.set_normalization(tokenization_settings["NUMERIC_TOKENIZATION"]["NORMALIZATION"])
    standard_deviation = encoder.set_standard_deviation(tokenization_settings["NUMERIC_TOKENIZATION"]["STANDARD_DEVIATION"])
    number_dims = encoder.set_number_dims(tokenization_settings["NUMERIC_TOKENIZATION"]["NUMBER_DIMS"])
    column_encoding = encoder.needs_column_encoding(mode)
    df_node = pd.read_csv(conf['DATASETS_PATH'] + str(node) + '.csv')

    if encoder.needs_min_max_values(mode, buckets):
        min_value = df_node[column_name].min()
        max_value = df_node[column_name].max()

    column_name_vector = None
    if column_encoding:
        column_name_vector = encoder.text_to_vec(column_name, '', terms, tokenization_settings)[1]

    if buckets:
        counter = df_node[column_name].count()
        step_size = counter / 300
        bucket_index = 0
        last_term = None
        remaining_step = step_size
        ordered_series = df_node[column_name].sort_values()
        ordered_series = ordered_series.dropna()
        for res in ordered_series:
            term = str(int(res))
            if term is None:
                continue
            if last_term is None:
                last_term = term

            while remaining_step < 1 and last_term != term:
                bucket_index += 1
                remaining_step += step_size
                if bucket_index >= 300:
                    bucket_index = 299
                    break
            last_term = term
            remaining_step -= 1

            if vec_dict.get(term) is not None:  # Don't calculate vector twice for the same term
                continue

            if mode == 'unary' or mode == 'unary-column-centroid':
                vec = encoder.bucket_to_vec_unary(bucket_index, column_name_vector)
            elif mode == 'unary-gaussian':
                vec = encoder.bucket_to_vec_unary_gaussian(bucket_index)
            elif mode == 'unary-column-partial':
                vec = encoder.bucket_to_vec_unary_column_partial(bucket_index, column_name_vector)
            elif mode == 'unary-random-dim':
                vec = encoder.bucket_to_vec_unary_random_dim(bucket_index, column_name_vector)
            elif mode == 'one-hot' or mode == 'one-hot-column-centroid':
                vec = encoder.bucket_to_vec_one_hot(bucket_index, column_name_vector)
            elif mode == 'one-hot-gaussian':
                vec = encoder.bucket_to_vec_one_hot_gaussian(bucket_index)
            elif mode == 'we-regression':
                vec = encoder.num_to_vec_we_regression(float(term))
            elif mode == 'random':
                vec = encoder.generate_random_vec()

            vec_dict[term] = dict()
            vec_dict[term]['vector'] = vec
    else:  # Not buckets
        new_df_node = df_node.copy()
        new_df_node[column_name] = new_df_node[column_name].astype('object')
        series_node = new_df_node[column_name].drop_duplicates()
        for res in series_node:
            term = str(res)
            if term is None:
                continue

            num = float(term)
            if mode == 'unary' or mode == 'unary-column-centroid':
                vec = encoder.num_to_vec_unary(num, min_value, max_value, column_name_vector)
            elif mode == 'unary-gaussian':
                vec = encoder.num_to_vec_unary_gaussian(num, min_value, max_value)
            elif mode == 'unary-gaussian-fluent':
                vec = encoder.num_to_vec_unary_gaussian_fluent(num, min_value, max_value)
            elif mode == 'unary-column-partial':
                vec = encoder.num_to_vec_unary_column_partial(num, min_value, max_value, column_name_vector)
            elif mode == 'unary-random-dim':
                vec = encoder.num_to_vec_unary_random_dim(num, min_value, max_value, column_name_vector)
            elif mode == 'one-hot' or mode == 'one-hot-column-centroid':
                vec = encoder.num_to_vec_one_hot(num, min_value, max_value, column_name_vector)
            elif mode == 'one-hot-gaussian':
                vec = encoder.num_to_vec_one_hot_gaussian(num, min_value, max_value)
            elif mode == 'one-hot-gaussian-fluent':
                vec = encoder.num_to_vec_one_hot_gaussian_fluent(num, min_value, max_value)
            elif mode == 'we-regression':
                vec = encoder.num_to_vec_we_regression(num)
            elif mode == 'random':
                vec = encoder.generate_random_vec()

            vec_dict[term] = dict()
            vec_dict[term]['vector'] = vec


def get_row_groups(graph, conf):
    print("Row relation extraction started...")
    result = dict()
    for node in graph.nodes:
        columns = graph.nodes[node]['columns']
        types = graph.nodes[node]['types']
        df_node = pd.read_csv(conf['DATASETS_PATH'] + str(node) + '.csv')
        if type(columns) != list or type(types) != list:
            continue
        columns_types = zip(columns, types)
        for (col1, type1), (col2, type2) in combinations(columns_types, 2):
            rel_name = '%s.%s~%s.%s' % (node, col1, node, col2)
            print('Processing ', rel_name)
            element_count = df_node.shape[0]
            complete_query = {
                "SELECT": "%s,%s" % (col1, col2),
                "FROM": node
            }
            new_group = get_group(rel_name, 'relational', element_count, query=complete_query, data_type=(type1, type2))
            if rel_name in result:
                result[rel_name].append(new_group)
            else:
                result[rel_name] = [new_group]
    return result


def get_relation_groups(graph, conf):
    # Assumption: two tables are only direct related by one foreign key relation
    print("Table relation extraction started:")
    result = dict()
    for (node1, node2, attrs) in graph.edges.data():
        table1, table2 = node1, node2
        df_table1 = pd.read_csv(conf['DATASETS_PATH'] + str(table1) + '.csv')
        df_table2 = pd.read_csv(conf['DATASETS_PATH'] + str(table2) + '.csv')
        key_col1, key_col2 = attrs['col1'], attrs['col2']
        columns_attr1 = graph.nodes[node1]['columns']
        column_names1 = columns_attr1 if type(columns_attr1) == list else [columns_attr1]
        columns_attr2 = graph.nodes[node2]['columns']
        column_names2 = columns_attr2 if type(columns_attr2) == list else [columns_attr2]
        types_attr1 = graph.nodes[node1]['types']
        types1 = types_attr1 if type(types_attr1) == list else [types_attr1]
        types_attr2 = graph.nodes[node2]['types']
        types2 = types_attr2 if type(types_attr2) == list else [types_attr2]
        for (col1, type1) in zip(column_names1, types1):
            for (col2, type2) in zip(column_names2, types2):
                print('Processing %s.%s~%s.%s ...' % (node1, col1, node2, col2))
                # Connect source with target
                count = 0
                rel_name = '%s.%s~%s.%s' % (node1, col1, node2, col2)
                if attrs['name'] == '-':
                    merging = pd.merge(df_table1, df_table2, left_on=key_col1, right_on=key_col2)
                    # Construct complete query for reconstruction
                    complete_query = {
                        "SELECT": "%s,%s" % (col1, col2),
                        "FROM": table1,
                        "JOIN": table2,
                        "LEFT_ON": key_col1,
                        "RIGHT_ON": key_col2
                    }
                else:
                    pkey_col1 = graph.nodes[node1]['pkey']
                    pkey_col2 = graph.nodes[node2]['pkey']
                    rel_tab_name = attrs['name']
                    df_rel_tab = pd.read_csv(conf['DATASETS_PATH'] + str(rel_tab_name) + '.csv')
                    merge1 = pd.merge(df_table1, df_rel_tab, left_on=pkey_col1, right_on=key_col1)
                    merging = pd.merge(merge1, df_table2, left_on=key_col2, right_on=pkey_col2)
                    # Construct complete query for reconstruction
                    complete_query = {
                        "SELECT": "%s,%s" % (col1, col2),
                        "FROM": table1,
                        "JOIN": [rel_tab_name, table2],
                        "LEFT_ON": [pkey_col1, key_col2],
                        "RIGHT_ON": [key_col1, pkey_col2]
                    }
                # Exclude numeric pair relations, to preserve values
                if not (type1 == "number" and type2 == "number"):
                    count = merging.shape[0]

                new_group = get_group(
                    attrs['name'], 'relational', count, query=complete_query, data_type=(type1, type2))
                if rel_name in result:
                    result[rel_name].append(new_group)
                else:
                    result[rel_name] = [new_group]
    return result


def output_groups(groups, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(groups, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def update_groups(groups, new_groups):
    for key in new_groups:
        if key in groups:
            groups[key] += new_groups[key]
        else:
            groups[key] = new_groups[key]
    return groups


def main(conf):
    print('Start loading graph...')
    graph = get_graph(path=conf['SCHEMA_GRAPH_PATH'])
    print('Retrieved graph data')

    groups = dict()
    we_table_name_path = conf['WE_ORIGINAL_TABLE_PATH']

    list_df = []
    counter = 1
    for chunk in pd.read_csv(we_table_name_path, chunksize=50000, na_filter=False):
        chunk['vector'] = chunk['vector'].apply(lambda x: x.replace('[', ''))
        chunk['vector'] = chunk['vector'].apply(lambda x: x.replace(']', ''))
        list_df.append(chunk)
        print(f'Process {counter * 50000} rows on GoogleVecs file')
        counter += 1

    df_vectors = pd.concat(list_df)
    df_vectors['id_vec'] = np.arange(1, len(df_vectors) + 1)

    # Get terms (like radix tree)
    terms = utils.get_terms_from_vector_set(df_vectors)

    # Get groups of values occurring in the same column
    groups = update_groups(groups, get_column_groups(df_vectors, graph, terms, conf))

    # Get all relations between text values in two columns in the same table
    groups = update_groups(groups, get_row_groups(graph, conf))

    # Get all relations in the graph
    groups = update_groups(groups, get_relation_groups(graph, conf))

    # Export groups
    print('Export groups ...')
    output_groups(groups, conf['GROUPS_FILE_NAME'])
