#!/usr/bin/python3

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn import preprocessing
from os import linesep

import RETRO.retro_utils as utils

MIN_GROUP_SIZE = 2


def get_adjacency_vector(size, group_name, index_lookup, conf):
    # get group elements (elements of column)
    print('Get adjacency vector for group:', group_name)
    table_name, column_name = utils.get_column_data_from_label(group_name, 'column')
    df = pd.read_csv(conf['DATASETS_PATH'] + table_name + '.csv')
    df = df[:conf['MAX_ROWS']]
    res = df[column_name].dropna()
    group_elements = [group_name + '#' + utils.tokenize(x) for idx, x in res.iteritems()]
    # construct vector
    vector = np.zeros(size)
    for element in group_elements:
        i = index_lookup[element]
        vector[i] = 1
    return vector


def retrieve_relation_elements(query, conf):
    if '|' in query:
        # Select with INNER JOIN
        if query.count('|') == 1:
            columns = []
            # One single JOIN
            selection, join = query.split('|')
            values = selection.split(',')
            for val in values:
                tab, col = val.split('.')
                columns.append(col)
            table1, table2, left_on, right_on = join.split(',')
            df_table1 = pd.read_csv(conf['DATASETS_PATH'] + table1 + '.csv')
            df_table1 = df_table1[:conf['MAX_ROWS']]
            df_table2 = pd.read_csv(conf['DATASETS_PATH'] + table2 + '.csv')
            df_table2 = df_table2[:conf['MAX_ROWS']]
            column_names1 = df_table1.columns
            column_names2 = df_table2.columns
            list1_as_set = set(column_names1)
            intersection = list1_as_set.intersection(column_names2)
            intersection_as_list = list(intersection)
            merge = pd.merge(df_table1, df_table2, left_on=left_on, right_on=right_on)
            if (columns[0] in intersection_as_list) and (columns[1] in intersection_as_list):
                group_elements = merge[[columns[0]+'_x', columns[1]+'_y']].dropna()
            elif (columns[0] not in intersection_as_list) and (columns[1] not in intersection_as_list):
                group_elements = merge[[columns[0], columns[1]]].dropna()
            elif (columns[0] in intersection_as_list) and (columns[1] not in intersection_as_list):
                group_elements = merge[[columns[0]+'_x', columns[1]]].dropna()
            else:
                group_elements = merge[[columns[0], columns[1]+'_y']].dropna()
        else:
            columns = []
            # Two JOIN
            selection, join1, join2 = query.split('|')
            values = selection.split(',')
            for val in values:
                tab, col = val.split('.')
                columns.append(col)
            table1, table2, left_on1, right_on1 = join1.split(',')
            _, table3, left_on2, right_on2 = join1.split(',')
            df_table1 = pd.read_csv(conf['DATASETS_PATH'] + table1 + '.csv')
            df_table1 = df_table1[:conf['MAX_ROWS']]
            df_table2 = pd.read_csv(conf['DATASETS_PATH'] + table2 + '.csv')
            df_table2 = df_table2[:conf['MAX_ROWS']]
            df_table3 = pd.read_csv(conf['DATASETS_PATH'] + table3 + '.csv')
            df_table3 = df_table3[:conf['MAX_ROWS']]
            merge1 = pd.merge(df_table1, df_table2, left_on=left_on1, right_on=right_on1)
            merge2 = pd.merge(merge1, df_table3, left_on=left_on2, right_on=right_on2)
            group_elements = merge2[columns].dropna()
    else:
        # Single Select
        columns = []
        values = query.split(',')
        for val in values:
            tab, col = val.split('.')
            table_name = tab
            columns.append(col)
        df = pd.read_csv(conf['DATASETS_PATH'] + table_name + '.csv')
        df = df[:conf['MAX_ROWS']]
        group_elements = df[columns].dropna()
    return group_elements


def fill_adjacency_matrix_relational(size, column_names, index_lookup, query, conf):
    # retrieve all relation elements from the database
    group_elements = retrieve_relation_elements(query, conf)

    # construct matrix and count vector
    A = lil_matrix((size, size))
    c_out = np.zeros(size)
    c_in = np.zeros(size)

    for (_, text_value1, text_value2) in group_elements.itertuples(name=None):
        text_value1 = utils.tokenize(text_value1)
        text_value2 = utils.tokenize(text_value2)
        i = index_lookup[utils.get_label(column_names[0], text_value1)]
        j = index_lookup[utils.get_label(column_names[1], text_value2)]
        A[i, j] = 1
        c_out[i] = 1
        c_in[j] = 1
    return csr_matrix(A), (c_in + c_out)


def create_adjacency_matrices(term_list, groups, conf):
    A_rel = dict()  # dict of (sparse) matrices or adj_vecs
    A_cat = dict()
    S = dict()  # dict of (sparse) matrices
    rel_key_pairs = set()  # set of pairs of inverse relations
    size = len(term_list)  # m' size of M0
    c = np.zeros(size)  # store for each vector how many relations it has
    # get lookup from matrix indices to terms
    index_lookup = utils.construct_index_lookup(term_list)
    # construct adjacency matrices from group information and db
    # relations of vectors were no word embeddings exist are missing in groups??
    for key in groups:
        for group in groups[key]:
            print('Process group %s:%s ...' % (key, group['name']), '(size: %d)' % (len(group['elements'])))
            matrix_key = '%s:%s' % (key, group['name'])
            suffix = ''
            if matrix_key in A_rel:
                suffix = str(len(groups[key]))
                matrix_key += suffix
            if group['type'] == 'categorial':
                A_cat[matrix_key] = get_adjacency_vector(size, key, index_lookup, conf)
                c += A_cat[matrix_key]
            if group['type'] == 'relational':
                if len(group['elements'].keys()) < MIN_GROUP_SIZE:
                    continue  # group is too small
                c1_t, c1_c, c2_t, c2_c = utils.get_column_data_from_label(key, 'relation')
                column1 = '%s.%s' % (c1_t, c1_c)
                column2 = '%s.%s' % (c2_t, c2_c)
                A_rel[matrix_key], c_inc = fill_adjacency_matrix_relational(
                    size, (column1, column2), index_lookup, group['query'], conf)
                reverse_key = '%s.%s~%s.%s:%s' % (c2_t, c2_c, c1_t, c1_c, group['name']) + suffix
                A_rel[reverse_key] = A_rel[matrix_key].T
                S[matrix_key] = preprocessing.normalize(A_rel[matrix_key], norm='l1')
                S[reverse_key] = preprocessing.normalize(A_rel[reverse_key], norm='l1')
                rel_key_pairs.add((matrix_key, reverse_key))
                c += c_inc  # c_inc = c_in + c_out
    return A_cat, S, c, rel_key_pairs


def create_M0(all_terms, present_vectors, dim):
    term_list = []
    M0 = []
    presence_vector = []
    m, s = utils.get_dist_params(present_vectors)
    for key in all_terms:
        for term in all_terms[key]:
            row_label = utils.get_label(key, term)
            term_list.append(row_label)
            if term in present_vectors[key]:
                presence_vector.append(1)
                M0.append(present_vectors[key][term])
            else:
                presence_vector.append(0)
                M0.append(np.zeros(dim))
    return term_list, np.array(M0), np.array(presence_vector)


def get_categorial_vector(v_cat, M0, presence_vector):
    v_norm = v_cat * presence_vector
    length = np.linalg.norm(v_norm, ord=1)
    if length > 0:
        v_norm = v_norm / np.linalg.norm(v_norm, ord=1)
        res = v_norm.dot(M0)
        return res
    else:
        return np.zeros(M0.shape[1])


def get_v_c(A_cat,  M0, presence_vector):  # parallel version
    res_cat = dict()
    for key in A_cat:
        res_cat[key] = get_categorial_vector(A_cat[key], M0, presence_vector)
    return res_cat


def calculate_Madd_rel(S, key, inv_key, c_inv, M_last, M_sum, v_denominator, conf):
    GAMMA = conf['GAMMA']
    DELTA = conf['DELTA']
    start = time.time()
    num_sources = len(set(S[key].nonzero()[0]))
    num_targets = len(set(S[inv_key].nonzero()[0]))
    max_cardinality = max(num_targets, num_sources)
    inv_nz = set(S[inv_key].nonzero()[0])
    targets_one_hot = np.array([1 if n in inv_nz else 0 for n in range(M_last.shape[0])])
    nz = set(S[key].nonzero()[0])
    sources_one_hot = np.array([1 if n in nz else 0 for n in range(M_last.shape[0])])
    max_c_inv = 1
    for i in range(M_last.shape[0]):
        if (sources_one_hot[i] != 0) and (c_inv[i] < max_c_inv):
            max_c_inv = c_inv[i]
        if (targets_one_hot[i] != 0) and (c_inv[i] < max_c_inv):
            max_c_inv = c_inv[i]
    gamma_i = np.zeros(M_last.shape[0])
    gamma_i_inv = list()
    gamma_i = np.array(S[key].max(axis=1).todense()).T[0]
    for i in gamma_i:
        gamma_i_inv.append(1 / i if i > 0 else 0)
    gamma_i_inv = np.array(gamma_i_inv)
    targets_weighted = (targets_one_hot * (1 / (max_cardinality * max_c_inv**-1))).dot(M_last)
    print('Preprocessing for', key, '--> done')

    M_inc = S[key].T.multiply(c_inv).T.dot(M_last) * GAMMA
    M_inc += (S[inv_key].T.multiply(c_inv).T).T.dot(M_last) * GAMMA
    M_dec = (targets_weighted * np.array([sources_one_hot]).T - S[key].dot(M_last) * gamma_i_inv[:, None] / (
        max_cardinality * max_c_inv**-1)) * DELTA * 2
    denum_sum = np.array(np.sum(S[inv_key].T.multiply(c_inv), axis=1)).T[0] * GAMMA + GAMMA * sources_one_hot * c_inv - 2 * DELTA * (
        num_targets - gamma_i_inv) / (max_cardinality * max_c_inv**-1) * sources_one_hot  # add to denominator
    M_sum += (M_inc - M_dec)
    v_denominator += denum_sum
    end = time.time()
    print('Calculation for relation', key, '--> done', 'time:', end - start)
    return


def calculate_Madd_cat(A_cat, key, Mc, M_sum, v_denominator, v_c, conf):
    BETA = conf['BETA']
    start = time.time()
    centroid = v_c[key]
    M_sum += np.array([Mc.diagonal() * A_cat[key]]).T * centroid * BETA
    v_denominator += Mc.diagonal() * A_cat[key] * BETA
    end = time.time()
    print('Calculation for categorial relation', key, '--> done', 'time:', end - start)
    return


def calculate_Mk(M0, M_last, Mc, c_inv, S, v_c, v_P, A_cat, invert_rel, term_list, conf):
    ALPHA = conf['ALPHA']
    M_sum = np.zeros(M0.shape)
    v_denominator = np.zeros(M0.shape[0], dtype='float32')
    M_sum += M0 * np.array([v_P]).T * ALPHA
    for key in A_cat:
        calculate_Madd_cat(A_cat, key, Mc, M_sum, v_denominator, v_c, conf)
    for key in S:
        calculate_Madd_rel(S, key, invert_rel[key], c_inv, M_last, M_sum, v_denominator, conf)

    result = M_sum / ((v_P * ALPHA + v_denominator)[:, None])
    print('Delta:', sum(np.linalg.norm(result - M_last, ord=1, axis=1)))
    print('Sum M_last', sum(np.linalg.norm(M_last, ord=1, axis=1)))
    print('Sum', sum(np.linalg.norm(result, ord=1, axis=1)))
    return result


def run_retrofitting(M0, S, v_c, c, v_P, A_cat, rel_key_pairs, term_list, conf):
    num_iter = conf['ITERATIONS'] if ('ITERATIONS' in conf) else 10
    size = M0.shape[0]
    # create matrix from c
    M_c = lil_matrix((size, size))
    c_inv = list()
    for elem in c:
        if elem > 0:
            c_inv.append(1 / elem)
        else:
            c_inv.append(0)
    c_inv = np.array(c_inv)
    M_c.setdiag(c_inv)

    # get lookup for rel_key_pairs
    invert_rel = dict()
    for (key1, key2) in rel_key_pairs:
        invert_rel[key1] = key2
        invert_rel[key2] = key1

    # call calculate_Mk in a loop
    Mk = np.copy(M0)
    for i in range(num_iter):
        print('Start Iteration: ', i)
        Mk = calculate_Mk(M0, Mk, M_c, c_inv, S, v_c, v_P, A_cat, invert_rel, term_list, conf)
    return Mk


def output_vectors(term_list, Mk, output_file_name):
    counter = 0
    for i, term in enumerate(term_list):
        is_all_zero = np.all((Mk[i] == 0))
        if not is_all_zero:
            counter += 1

    # init output file
    f_out = open(output_file_name, 'w')
    # write meta information
    f_out.write('%d %d' % (counter, Mk.shape[1]) + linesep)
    # write term vector pairs
    for i, term in enumerate(term_list):
        is_all_zero = np.all((Mk[i] == 0))
        if not is_all_zero:
            print('Exported', i, 'term vectors | Current term:', term)
            f_out.write('%s %s' % (term, ' '.join([str(x) for x in Mk[i]])))
            f_out.write(linesep)
    f_out.close()
    return


def main(conf):
    # get group information
    groups_info = utils.parse_groups(conf['GROUPS_FILE_NAME'])
    data_columns = utils.get_data_columns_from_group_data(groups_info)

    # get tokens of data columns
    all_terms = utils.get_terms(data_columns, conf)
    print('Retrieved terms from database')

    present_vectors, dim = utils.get_vectors_for_present_terms_from_group_file(data_columns, groups_info)
    print('Got vectors of terms from group file')

    # create M0 and presence vector
    term_list, M0, v_P = create_M0(all_terms, present_vectors, dim)
    print('Constructed initial matrix M0 with size', M0.shape)
    print('Len', len(v_P.nonzero()[0]))

    # create adjacency matrices, weight matrices, count vectors and vector for R
    A_cat, S, c, rel_key_pairs = create_adjacency_matrices(term_list, groups_info, conf)
    print('Created matrix representations')

    # get category vectors v_c
    v_c = get_v_c(A_cat, M0, v_P)
    for key in v_c:
        print(key, np.linalg.norm(v_c[key]))

    print('Created category vectors')
    v_Q = np.ones(len(v_P))
    # run iterative algorithm
    Mk = run_retrofitting(M0, S, v_c, c, v_Q, A_cat, rel_key_pairs, term_list, conf)
    print('Retrofitting done, start to generate vectors file ...')

    # output result to file
    output_vectors(term_list, Mk, conf['RETRO_VECS_FILE_NAME'])
    print('Exported vectors')

    return
