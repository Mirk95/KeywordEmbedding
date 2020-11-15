import re
import pickle
import numpy as np
import pandas as pd


def parse_groups(group_filename):
    with open(group_filename, 'rb') as handle:
        groups = pickle.load(handle)
    for key in groups:
        for group in groups[key]:
            if group['type'] == 'categorial':
                for key in group['elements']:
                    group['elements'][key]['vector'] = group['elements'][key]['vector']
            if 'inferred_elements' in group:
                if group['type'] == 'categorial':
                    for key in group['inferred_elements']:
                        group['inferred_elements'][key]['vector'] = group['inferred_elements'][key]['vector']
    return groups


def get_data_columns_from_group_data(groups):
    result = set()
    for key in groups:
        if groups[key][0]['type'] == 'categorial':
            result.add((key, groups[key][0]['data_type']))
    return list(result)


def get_column_data_from_label(label, type):
    if type == 'column':
        try:
            table_name, column_name = label.split('.')
            return table_name, column_name
        except ValueError:
            print('ERROR: Can not decode %s into table name and column name' % label)
            return
    if type == 'relation':
        try:
            c1, c2 = label.split('~')
            c1_table_name, c1_column_name = c1.split('.')
            c2_table_name, c2_column_name = c2.split('.')
            return c1_table_name, c1_column_name, c2_table_name, c2_column_name
        except ValueError:
            print('ERROR: Can not decode relation label %s ' % label)
            return


def get_label(x, y): return '%s#%s' % (x, y)


def tokenize(term):
    if type(term) == str:
        return re.sub('[\.#~\s,\(\)/\[\]:]+', '_', str(term))
    else:
        return ''


def get_terms(columns, conf):
    result = dict()
    for column, data_type in columns:
        table_name, column_name = column.split('.')
        df = pd.read_csv(conf['DATASETS_PATH'] + table_name + '.csv')
        res = df[column_name]
        if data_type == 'number':
            res = res.fillna(np.nan).replace([np.nan], [None])
            result[column] = [str(int(x)) if x is not None else x for idx, x in res.iteritems()]
        else:
            res = res.replace('nan', '')
            res = res.replace(np.nan, '')
            result[column] = [tokenize(x) for idx, x in res.iteritems()]
        result[column] = list(set(result[column]))  # Remove duplicates
    return result


def construct_index_lookup(list_obj):
    result = dict()
    for i in range(len(list_obj)):
        result[list_obj[i]] = i
    return result


def execute_threads_from_pool(thread_pool, verbose=False):
    while len(thread_pool) > 0:
        try:
            next = thread_pool.pop()
            if verbose:
                print('Number of threads:', len(thread_pool))
            next.start()
            next.join()
        except ValueError:
            print("Warning: threadpool.pop() failed")
    return


def get_vectors_for_present_terms_from_group_file(data_columns, groups_info):
    result_present = dict()
    dim = 0
    for column, data_type in data_columns:
        group = groups_info[column][0]['elements']
        group_extended = groups_info[column][0]['inferred_elements']
        result_present[column] = dict()
        for term in group:
            result_present[column][term] = np.array(group[term]['vector'], dtype='float32')
            dim = len(result_present[column][term])
        for term in group_extended:
            result_present[column][term] = np.array(group_extended[term]['vector'], dtype='float32')
            dim = len(result_present[column][term])
    return result_present, dim


def get_terms_from_vector_set(df_vectors):
    print("Getting terms from vector table:")
    chunk_size = 100000
    term_dict = dict()
    min_id = 0
    max_id = chunk_size
    while True:
        print("%s to %s..." % (min_id, max_id))
        records = df_vectors[min_id:max_id].to_records(index=False)
        term_list = list(records)
        if len(term_list) < 1:
            break
        for (term, vector, freq) in term_list:
            term = str(term)
            freq = int(freq)
            splits = term.split('_')
            current = [term_dict, None, -1]
            i = 1
            while i <= len(splits):
                sub_term = '_'.join(splits[:i])
                if sub_term in current[0]:
                    current = current[0][sub_term]
                else:
                    current[0][sub_term] = [dict(), None, -1]
                    current = current[0][sub_term]
                i += 1
            current[1] = vector
            current[2] = freq
        if max_id >= 100000:
            break
        min_id = max_id
        max_id += chunk_size
    return term_dict
