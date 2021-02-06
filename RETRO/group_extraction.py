#!/usr/bin/python3

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

import RETRO.retro_utils as utils


def get_graph(path, graph_type='gml'):
    g = None
    if graph_type == 'gml':
        g = nx.read_gml(path)
    return g


def get_group(name, group_type, vector_dict, extended=None, query='', export_type='full'):
    elements = []
    if group_type == 'categorial':
        if export_type == 'full':
            elements = vector_dict
        else:
            elements = list(vector_dict.keys())
    else:
        if export_type == 'full':
            elements = vector_dict
        else:
            elements = list(vector_dict.keys())
    result = {
        'name': name,
        'type': group_type,
        'elements': elements,
        'query': query
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
    for node in graph.nodes:
        columns_attr = graph.nodes[node]['columns']
        column_names = columns_attr if type(columns_attr) == list else [columns_attr]
        df_node = pd.read_csv(conf['DATASETS_PATH'] + str(node) + '.csv', na_filter=False)
        df_node = df_node.applymap(str)
        df_node = df_node.applymap(lambda x: utils.tokenize(x) if isinstance(x, str) else x)
        for column_name in column_names:
            print('Process %s.%s ...' % (node, column_name))
            vec_dict_fit = dict()
            vec_dict_inferred = dict()
            merging = pd.merge(df_node, df_vectors, how='left', left_on=column_name, right_on=df_vectors.word)
            merging = merging[[column_name, 'vector', 'id_vec']]
            merging = merging.fillna('')
            records = merging.to_records(index=False)
            term_vecs = list(records)
            for (term, vec_bytes, vec_id) in term_vecs:
                # Modified the following rows to support multi-words per token
                for val in term.split('_'):
                    if vec_bytes != '':
                        vec_dict_fit[val] = dict()
                        vec_dict_fit[val]['vector'] = np.array(vec_bytes.split(), dtype='float32')
                        vec_dict_fit[val]['id'] = int(vec_id)
                    else:
                        if val == '':
                            continue
                        splits = [x.replace('_', '') for x in val.split('_')]
                        i = 1
                        j = 0
                        current = [terms, None, -1]
                        vector = None
                        last_match = (0, None, -1)
                        count = 0
                        while i <= len(splits) or last_match[1] is not None:
                            sub_word = '_'.join(splits[j:i])
                            if sub_word in current[0]:
                                current = current[0][sub_word]
                                if (current[1] != '') and (current[1] is not None):
                                    last_match = (i, np.array(current[1].split(), dtype='float32'), current[2])
                            else:
                                if last_match[1] is not None:
                                    if vector is not None:
                                        if conf['TOKENIZATION'] == 'log10':
                                            vector += last_match[1] * np.log10(last_match[2])
                                            count += np.log10(last_match[2])
                                        else:  # 'simple' or different
                                            vector += last_match[1]
                                            count += 1
                                    else:
                                        if conf['TOKENIZATION'] == 'log10':
                                            vector = last_match[1] * np.log10(last_match[2])
                                            count += np.log10(last_match[2])
                                        else:  # 'simple' or different
                                            vector = last_match[1]
                                            count += 1
                                    j = last_match[0]
                                    i = j
                                    last_match = (0, None, -1)
                                else:
                                    j += 1
                                    i = j
                                current = [terms, None, -1]
                            i += 1
                        if vector is not None:
                            vector /= count
                            vec_dict_inferred[val] = dict()
                            vec_dict_inferred[val]['vector'] = vector
            result['%s.%s' % (node, column_name)] = [get_group('%s.%s' % (node, column_name),
                                                               'categorial',
                                                               vec_dict_fit,
                                                               extended=vec_dict_inferred)]
    return result


def get_row_groups(df_vectors, graph, conf):
    print("Row relation extraction started...")
    result = dict()
    for node in graph.nodes:
        columns = graph.nodes[node]['columns']
        df_node = pd.read_csv(conf['DATASETS_PATH'] + str(node) + '.csv', na_filter=False)
        df_node = df_node.applymap(str)
        df_node = df_node.applymap(lambda x: utils.tokenize(x) if isinstance(x, str) else x)
        if type(columns) != list:
            continue
        for col1, col2 in combinations(columns, 2):
            vec_dict = dict()
            rel_name = '%s.%s~%s.%s' % (node, col1, node, col2)
            print('Processing ', rel_name)
            merge1 = pd.merge(df_node, df_vectors, how='inner', left_on=col1, right_on=df_vectors.word)
            merge2 = pd.merge(merge1, df_vectors, how='inner', left_on=col2, right_on=df_vectors.word)
            merging = merge2[[col1, col2, 'vector_x', 'vector_y', 'id_vec_x', 'id_vec_y']]
            merging = merging.fillna('')
            records = merging.to_records(index=False)
            res = list(records)
            complete_query = {
                "SELECT": "%s,%s" % (col1, col2),
                "FROM": node
            }
            for (term1, term2, vec1, vec2, id_vec1, id_vec2) in res:
                key = '%s~%s' % (term1, term2)
                vec_dict[key] = dict()
                vec_dict[key]['ids'] = [int(id_vec1), int(id_vec2)]
            new_group = get_group(rel_name, 'relational', vec_dict, query=complete_query)
            if rel_name in result:
                result[rel_name].append(new_group)
            else:
                result[rel_name] = [new_group]
    return result


def get_relation_groups(df_vectors, graph, conf):
    # Assumption: two tables are only direct related by one foreign key relation
    print("Table relation extraction started:")
    result = dict()
    for (node1, node2, attrs) in graph.edges.data():
        table1, table2 = node1, node2
        df_table1 = pd.read_csv(conf['DATASETS_PATH'] + str(table1) + '.csv', na_filter=False)
        df_table1 = df_table1.applymap(str)
        df_table1 = df_table1.applymap(lambda x: utils.tokenize(x) if isinstance(x, str) else x)
        df_table2 = pd.read_csv(conf['DATASETS_PATH'] + str(table2) + '.csv', na_filter=False)
        df_table2 = df_table2.applymap(str)
        df_table2 = df_table2.applymap(lambda x: utils.tokenize(x) if isinstance(x, str) else x)
        key_col1, key_col2 = attrs['col1'], attrs['col2']
        columns_attr1 = graph.nodes[node1]['columns']
        column_names1 = columns_attr1 if type(columns_attr1) == list else [columns_attr1]
        columns_attr2 = graph.nodes[node2]['columns']
        column_names2 = columns_attr2 if type(columns_attr2) == list else [columns_attr2]
        list1_as_set = set(column_names1)
        intersection = list1_as_set.intersection(column_names2)
        intersection_as_list = list(intersection)
        for col1 in column_names1:
            for col2 in column_names2:
                print('Process %s.%s~%s.%s ...' % (node1, col1, node2, col2))
                # Connect source with target
                vec_dict = dict()
                rel_name = '%s.%s~%s.%s' % (node1, col1, node2, col2)
                if attrs['name'] == '-':
                    merge1 = pd.merge(df_table1, df_table2, left_on=key_col1, right_on=key_col2)
                    if (col1 in intersection_as_list) and (col2 in intersection_as_list):
                        merge2 = pd.merge(merge1, df_vectors, left_on=col1 + '_x', right_on=df_vectors.word)
                        merge3 = pd.merge(merge2, df_vectors, left_on=col2 + '_y', right_on=df_vectors.word)
                        merging = merge3[[col1 + '_x', col2 + '_y', 'vector_x', 'vector_y', 'id_vec_x', 'id_vec_y']]
                    elif (col1 not in intersection_as_list) and (col2 not in intersection_as_list):
                        merge2 = pd.merge(merge1, df_vectors, left_on=col1, right_on=df_vectors.word)
                        merge3 = pd.merge(merge2, df_vectors, left_on=col2, right_on=df_vectors.word)
                        merging = merge3[[col1, col2, 'vector_x', 'vector_y', 'id_vec_x', 'id_vec_y']]
                    elif (col1 in intersection_as_list) and (col2 not in intersection_as_list):
                        merge2 = pd.merge(merge1, df_vectors, left_on=col1 + '_x', right_on=df_vectors.word)
                        merge3 = pd.merge(merge2, df_vectors, left_on=col2, right_on=df_vectors.word)
                        merging = merge3[[col1 + '_x', col2, 'vector_x', 'vector_y', 'id_vec_x', 'id_vec_y']]
                    else:
                        merge2 = pd.merge(merge1, df_vectors, left_on=col1, right_on=df_vectors.word)
                        merge3 = pd.merge(merge2, df_vectors, left_on=col2 + '_y', right_on=df_vectors.word)
                        merging = merge3[[col1, col2 + '_y', 'vector_x', 'vector_y', 'id_vec_x', 'id_vec_y']]
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
                    df_rel_tab = pd.read_csv(conf['DATASETS_PATH'] + str(rel_tab_name) + '.csv', na_filter=False)
                    df_rel_tab = df_rel_tab.applymap(str)
                    df_rel_tab = df_rel_tab.applymap(lambda x: utils.tokenize(x) if isinstance(x, str) else x)
                    merge1 = pd.merge(df_table1, df_rel_tab, left_on=pkey_col1, right_on=key_col1)
                    merge2 = pd.merge(merge1, df_table2, left_on=key_col2, right_on=pkey_col2)
                    merge3 = pd.merge(merge2, df_vectors, left_on=col1, right_on=df_vectors.word)
                    merge4 = pd.merge(merge3, df_vectors, left_on=col2, right_on=df_vectors.word)
                    merging = merge4[[col1, col2, 'vector_x', 'vector_y', 'id_vec_x', 'id_vec_y']]
                    # Construct complete query for reconstruction
                    complete_query = {
                        "SELECT": "%s,%s" % (col1, col2),
                        "FROM": table1,
                        "JOIN": [rel_tab_name, table2],
                        "LEFT_ON": [pkey_col1, key_col2],
                        "RIGHT_ON": [key_col1, pkey_col2]
                    }
                merging = merging.fillna('')
                records = merging.to_records(index=False)
                res = list(records)
                for (term1, term2, vec1_bytes, vec2_bytes, vec1_id, vec2_id) in res:
                    key = '%s~%s' % (term1, term2)
                    vec_dict[key] = dict()
                    vec_dict[key]['ids'] = [int(vec1_id), int(vec2_id)]
                new_group = get_group(attrs['name'], 'relational', vec_dict, query=complete_query)
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

    # Get groups of text values occurring in the same column
    groups = update_groups(groups, get_column_groups(df_vectors, graph, terms, conf))

    # Get all relations between text values in two columns in the same table
    groups = update_groups(groups, get_row_groups(df_vectors, graph, conf))

    # Get all relations in the graph
    groups = update_groups(groups, get_relation_groups(df_vectors, graph, conf))

    # Export groups
    print('Export groups ...')
    output_groups(groups, conf['GROUPS_FILE_NAME'])
