import json
import numpy as np
from datetime import datetime

REFERENCE_EMBEDDING_TABLE = 'google_vecs'
RETROFITTED_EMBEDDING_TABLE = 'retro_vecs'

GROUPS_KEY = 'GROUPS_FILE_NAME'

def parse_bin_vec(data):
    vec =  np.fromstring(bytes(data), dtype='float32')
    if np.linalg.norm(vec) < 0.001:
        vec += np.ones(vec.shape[0])*0.001
    return vec

def get_db_vectors(table_name, con, cur, freq_limit=0, return_frequencies=False):
    result = dict()
    filtered = dict()
    query = "SELECT id, word, vector FROM %s" % (table_name,)
    cur.execute(query)

    for (id, term, vec_bytes) in cur.fetchall():
        if int(id) > freq_limit:
            if (return_frequencies):
                result[term] = (np.fromstring(bytes(vec_bytes), dtype='float32'), id)
            else:
                result[term] = np.fromstring(bytes(vec_bytes), dtype='float32')
        else:
            if (return_frequencies):
                filtered[term] = (np.fromstring(bytes(vec_bytes), dtype='float32'), id)
            else:
                filtered[term] = np.fromstring(bytes(vec_bytes), dtype='float32')
    if freq_limit != 0:
        return result, filtered
    else:
        return result

def vec_dict2matrix(vecs, max_size=10000000):
    data = []
    key_data = list(vecs.keys())
    key_data = key_data[:max_size]
    for i in range(len(key_data)):
        data.append(vecs[key_data[i]])
    data = np.array(data)
    return key_data, data


def get_retro_vecs(con, cur, table_name=RETROFITTED_EMBEDDING_TABLE):
    return get_db_vectors(table_name, con, cur)

def create_outliers_lookup(outliers_file_name):
    result = dict()
    f = open(outliers_file_name, 'r')
    json_data = json.load(f)
    for key in json_data:
        result[key] = json_data[key]['outliers']
    return result

def get_vector_query(query_tmpl, table_names, vec_table_templ, vec_table1_templ, vec_table2_templ):
    query = ""
    if len(table_names) == 1:
        query = query_tmpl.replace(vec_table_templ, table_names[0])
    else:
        query = query_tmpl.replace(vec_table1_templ, table_names[0])
        query = query.replace(vec_table2_templ, table_names[1])
    return query

def create_timestamp():
    return str(datetime.now().timestamp())
