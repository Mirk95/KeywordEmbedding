import re
import pickle
import numpy as np
import pandas as pd


def parse_groups(group_filename):
    with open(group_filename, 'rb') as handle:
        groups = pickle.load(handle)
    for key in groups:
        for group in groups[key]:
            for key in group['elements']:
                if group['type'] == 'categorial':
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
            result.add(key)
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
    for column in columns:
        table_name, column_name = column.split('.')
        df = pd.read_csv(conf['DATASETS_PATH'] + table_name + '.csv')
        df = df[:conf['MAX_ROWS']]
        res = df[column_name].dropna()
        result[column] = [tokenize(x) for idx, x in res.iteritems()]
        result[column] = list(set(result[column]))  # remove duplicates
    return result


def construct_index_lookup(list_obj):
    result = dict()
    for i in range(len(list_obj)):
        result[list_obj[i]] = i
    return result


def get_dist_params(vectors):
    # returns the distribution parameter for vector elements
    m_value = 0
    count = 0
    values = []
    for key in vectors:
        max_inst = 0
        for term in vectors[key]:
            m_value += np.mean(vectors[key][term])
            values.extend([x for x in vectors[key][term]])
            max_inst += 1
            count += 1
            if max_inst > 100:
                break
    m_value /= count
    s_value = np.mean((np.array(values) - m_value)**2)
    return m_value, s_value


def get_vectors_for_present_terms_from_group_file(data_columns, groups_info):
    result_present = dict()
    dim = 0
    for column in data_columns:
        group = groups_info[column][0]['elements']
        group_extended = groups_info[column][0]['inferred_elements']
        result_present[column] = dict()
        for term in group:
            result_present[column][term] = np.array(group[term]['vector'].split(), dtype='float32')
            dim = len(result_present[column][term])
        for term in group_extended:
            result_present[column][term] = np.array(group_extended[term]['vector'], dtype='float32')
            dim = len(result_present[column][term])
    return result_present, dim


# UPDATED function
def get_terms_from_vector_set(conf):
    print("Getting terms from vector table:")
    chunk_size = 500000
    list_df = []
    term_dict = dict()
    min_id = 0
    max_id = chunk_size
    for chunk in pd.read_csv(conf['WE_ORIGINAL_TABLE_PATH'], chunksize=chunk_size):
        chunk['vector'] = chunk['vector'].apply(lambda x: x.replace('[', ''))
        chunk['vector'] = chunk['vector'].apply(lambda x: x.replace(']', ''))
        list_df.append(chunk)
        term_list = []
        for idx, word, vector in chunk.itertuples(name=None):
            term = [str(word), vector, int(idx)]
            term_list.append(term)
        if len(term_list) < 1:
            break
        print("%s to %s..." % (min_id, max_id))
        for (term, vector, freq) in term_list:
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
        if max_id == conf['MAX_ROWS']:
            break
        min_id = max_id
        max_id += chunk_size
    df_vectors = pd.concat(list_df)
    return df_vectors, term_dict
