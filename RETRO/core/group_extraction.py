#!/usr/bin/python3

import json
import sys
import networkx as nx
from itertools import combinations
import base64

import config
import db_connection as db_con
import retro_utils as utils
import encoding_utils as encoder


def get_graph(path, graph_type='gml'):
    g = None
    if graph_type == 'gml':
        g = nx.read_gml(path)
    return g


def get_group(name, group_type, vector_dict, extended=None, query='', export_type='full', data_type='string'):
    elems = []
    if group_type == 'categorial':
        if export_type == 'full':
            elems = vector_dict
        else:
            elems = list(vector_dict.keys())
    else:
        elems = vector_dict  # vector_dict is just the count of the elements
    result = {
        'name': name,
        'type': group_type,
        'elements': elems,
        'query': query,
        'data_type': data_type  # 'string' or 'number'
    }
    if extended != None:
        if export_type == 'full':
            result['inferred_elements'] = extended
        else:
            result['inferred_elements'] = list(extended.keys())
    return result


def get_column_groups(graph, we_table_name, terms, con, cur, tokenization_settings):
    print("Column relation extraction started:")
    result = dict()
    # initialize tokenization algorithms
    initialize_numeric_tokenization(cur, we_table_name, tokenization_settings)

    # construct query
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
                get_numeric_column_groups(cur, node, column_name, vec_dict_inferred, we_table_name, terms,
                                          tokenization_settings)

            else:  # Process string values
                query = "SELECT %s, we.vector, we.id FROM %s LEFT OUTER JOIN %s AS we ON %s = we.word" % (
                    utils.tokenize_sql_variable('%s.%s' % (node, column_name)),
                    node, we_table_name,
                    utils.tokenize_sql_variable('%s.%s' % (node, column_name)))
                cur.execute(query)
                term_vecs = cur.fetchall()

                for (term, vec_bytes, vec_id) in term_vecs:
                    inferred, vector = encoder.text_to_vec(term, vec_bytes, terms, tokenization_settings)
                    if inferred:
                        if vector is None:
                            continue
                        vec_dict_inferred[term] = dict()
                        vec_dict_inferred[term]['vector'] = base64.encodebytes(vector).decode('ascii')
                    else:
                        vec_dict_fit[term] = dict()
                        vec_dict_fit[term]['vector'] = base64.encodebytes(vector).decode('ascii')
                        vec_dict_fit[term]['id'] = int(vec_id)

            result['%s.%s' % (node, column_name)] = [get_group(
                '%s.%s' % (node, column_name), 'categorial', vec_dict_fit, extended=vec_dict_inferred,
                data_type=column_type)]
            # here a clustering approach could be done
    return result


def initialize_numeric_tokenization(cur, we_table_name, tokenization_strategy):
    if tokenization_strategy == 'we-regression':
        encoder.initialize_numeric_word_embeddings(cur, we_table_name)


def get_numeric_column_groups(cur, table_name, column_name, vec_dict, we_table_name, terms, tokenization_settings):
    mode = tokenization_settings["NUMERIC_TOKENIZATION"]["MODE"]
    buckets = encoder.set_buckets(tokenization_settings["NUMERIC_TOKENIZATION"]["BUCKETS"])
    normalization = encoder.set_normalization(tokenization_settings["NUMERIC_TOKENIZATION"]["NORMALIZATION"])
    standard_deviation = encoder.set_standard_deviation(tokenization_settings["NUMERIC_TOKENIZATION"]["STANDARD_DEVIATION"])
    number_dims = encoder.set_number_dims(tokenization_settings["NUMERIC_TOKENIZATION"]["NUMBER_DIMS"])
    column_encoding = encoder.needs_column_encoding(mode)

    if encoder.needs_min_max_values(mode, buckets):
        min_value = 0
        max_value = 0
        min_query = "SELECT min(%s) FROM %s" % \
                    ('%s.%s' % (table_name, column_name), table_name)
        cur.execute(min_query)
        min_value = cur.fetchall()[0][0]
        max_query = "SELECT max(%s) FROM %s" % \
                    ('%s.%s' % (table_name, column_name), table_name)
        cur.execute(max_query)
        max_value = cur.fetchall()[0][0]

    column_name_vector = None
    if column_encoding:
        column_name_vector = encoder.text_to_vec(column_name, None, terms, tokenization_settings)[1]

    if buckets:
        count_query = "SELECT COUNT(%s) FROM %s" % ('%s.%s' % (table_name, column_name), table_name)
        cur.execute(count_query)
        count = cur.fetchall()[0][0]
        step_size = count / 300
        bucket_index = 0
        last_term = None
        remaining_step = step_size
        element_query = "SELECT  %s::varchar FROM %s ORDER BY %s" % \
                        ('%s.%s' % (table_name, column_name), table_name, '%s.%s' % (table_name, column_name))
        cur.execute(element_query)
        for res in cur.fetchall():
            term = res[0]
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

            if vec_dict.get(term) is not None:  # don't calculate vector twice for the same term
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
            vec_dict[term]['vector'] = base64.encodebytes(vec).decode('ascii')
    else:  # not buckets
        element_query = "SELECT DISTINCT %s::varchar FROM %s" % \
                        ('%s.%s' % (table_name, column_name), table_name)
        cur.execute(element_query)
        for res in cur.fetchall():
            term = res[0]
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
            vec_dict[term]['vector'] = base64.encodebytes(vec).decode('ascii')


def get_row_groups(graph, we_table_name, con, cur):
    print("Row relation extraction started...")
    result = dict()
    for node in graph.nodes:
        columns = graph.nodes[node]['columns']
        types = graph.nodes[node]['types']
        if type(columns) != list or type(types) != list:
            continue
        columns_types = zip(columns, types)
        for (col1, type1), (col2, type2) in combinations(columns_types, 2):
            rel_name = '%s.%s~%s.%s' % (node, col1, node, col2)
            print('Processing ', rel_name)
            col1_query_symbol = ('%s.%s' % (node, col1)) if type1 == "number" \
                else utils.tokenize_sql_variable('%s.%s' % (node, col1))
            col2_query_symbol = ('%s.%s' % (node, col2)) if type2 == "number" \
                else utils.tokenize_sql_variable('%s.%s' % (node, col2))
            element_count = 0
            count_query = "SELECT COUNT(*) FROM %s" % node  # returns element count
            cur.execute(count_query)
            element_count = cur.fetchall()[0][0]

            complete_query = "SELECT %s::varchar, %s::varchar FROM %s" % (col1_query_symbol, col2_query_symbol, node)
            new_group = get_group(rel_name, 'relational', element_count, query=complete_query, data_type=(type1, type2))
            if rel_name in result:
                result[rel_name].append(new_group)
            else:
                result[rel_name] = [new_group]
    return result


def get_relation_groups(graph, we_table_name, con, cur):
    # Assumption: two tables are only direct related by one foreign key relation
    print("Table relation extraction started:")
    result = dict()
    for (node1, node2, attrs) in graph.edges.data():
        table1, table2 = node1, node2
        key_col1, key_col2 = attrs['col1'], attrs['col2']
        columns_attr1 = graph.nodes[node1]['columns']
        column_names1 = columns_attr1 if type(columns_attr1) == list else [
            columns_attr1
        ]
        columns_attr2 = graph.nodes[node2]['columns']
        column_names2 = columns_attr2 if type(columns_attr2) == list else [
            columns_attr2
        ]
        types_attr1 = graph.nodes[node1]['types']
        types1 = types_attr1 if type(types_attr1) == list else [
            types_attr1
        ]
        types_attr2 = graph.nodes[node2]['types']
        types2 = types_attr2 if type(types_attr2) == list else [
            types_attr2
        ]
        for (col1, type1) in zip(column_names1, types1):
            col1_query_symbol = ('%s.%s' % (table1, col1)) if type1 == "number" \
                else utils.tokenize_sql_variable('%s.%s' % (table1, col1))

            for (col2, type2) in zip(column_names2, types2):
                print('Processing %s.%s~%s.%s ...' % (node1, col1, node2, col2))
                col2_query_symbol = ('%s.%s' % (table2, col2)) if type2 == "number" \
                    else utils.tokenize_sql_variable('%s.%s' % (table2, col2))
                # conect source with target
                rel_name = ''
                count = 0
                rel_name = '%s.%s~%s.%s' % (node1, col1, node2, col2)
                count_query = ''
                complete_query = ''
                if attrs['name'] == '-':
                    count_query = ("SELECT COUNT(*) "
                                + "FROM %s INNER JOIN %s ON %s.%s = %s.%s ") % (
                               table1, table2, table1, key_col1, table2, key_col2)  # returns term pair count
                    # construct complete query for reconstruction
                    complete_query = "SELECT %s::varchar, %s::varchar FROM %s INNER JOIN %s ON %s.%s = %s.%s " \
                                     % (col1_query_symbol,
                                        col2_query_symbol,
                                        table1, table2, table1, key_col1,
                                        table2, key_col2)
                else:
                    pkey_col1 = graph.nodes[node1]['pkey']
                    pkey_col2 = graph.nodes[node2]['pkey']
                    rel_tab_name = attrs['name']
                    count_query = ("SELECT COUNT(*) "
                                + "FROM %s INNER JOIN %s ON %s.%s = %s.%s "
                                + "INNER JOIN %s ON %s.%s = %s.%s ") % (
                                   table1, rel_tab_name, table1, pkey_col1, rel_tab_name, key_col1,
                                   table2, table2, pkey_col2, rel_tab_name, key_col2)  # returns (term1, term2)
                    # construct complete query for reconstruction
                    complete_query = ("SELECT %s::varchar, %s::varchar FROM %s " +
                                      "INNER JOIN %s ON %s.%s = %s.%s "
                                      + "INNER JOIN %s ON %s.%s = %s.%s") % (
                                         col1_query_symbol,
                                         col2_query_symbol, table1,
                                         rel_tab_name, table1, pkey_col1,
                                         rel_tab_name, key_col1, table2,
                                         table2, pkey_col2, rel_tab_name,
                                         key_col2)
                # Exclude numeric pair relations, to preserve values
                if not (type1 == "number" and type2 == "number"):
                    cur.execute(count_query)
                    count = cur.fetchall()[0][0]

                new_group = get_group(
                    attrs['name'], 'relational', count, query=complete_query, data_type=(type1, type2))
                if rel_name in result:
                    result[rel_name].append(new_group)
                else:
                    result[rel_name] = [new_group]
    return result


def output_groups(groups, filename):
    f = open(filename, 'w')
    f.write(json.dumps(groups))
    f.close()
    return


def update_groups(groups, new_groups):
    for key in new_groups:
        if key in groups:
            groups[key] += new_groups[key]
        else:
            groups[key] = new_groups[key]
    return groups


def main(argc, argv):
    db_config = db_con.get_db_config(path=argv[2])
    con, cur = db_con.create_connection(db_config)

    # get retrofitting config
    conf = config.get_config(argv)

    print('Start loading graph...')
    graph = get_graph(path=conf['SCHEMA_GRAPH_PATH'])
    print('Retrieved graph data')

    groups = dict()

    we_table_name = conf['WE_ORIGINAL_TABLE_NAME']

    # get terms (like radix tree)
    terms = utils.get_terms_from_vector_set(we_table_name, con, cur)

    # get groups of values occuring in the same column
    groups = update_groups(groups, get_column_groups(
        graph, we_table_name, terms, con, cur, conf['TOKENIZATION_SETTINGS']))

    # get all relations between text values in two columns in the same table
    groups = update_groups(groups, get_row_groups(
        graph, we_table_name, con, cur))

    # get all relations in the graph
    groups = update_groups(groups, get_relation_groups(
        graph, we_table_name, con, cur))

    # export groups
    print('Export groups ...')
    output_groups(groups, conf['GROUPS_FILE_NAME'])


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
