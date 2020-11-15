#!/usr/bin/python3

import os
import json
import networkx as nx
from collections import defaultdict
from itertools import combinations


def get_schema(schema_dir, vectors_filename):
    """
        :param schema_dir: path to schema directory
        :param vectors_filename: filename with pre-trained embeddings
        :return schema: a dictionary with pk and fk separated
        """
    rels = []
    db_schema_path = schema_dir + 'db_schema.json'
    vectors_schema_path = schema_dir + vectors_filename + '_schema.json'

    if os.path.isfile(db_schema_path) and os.path.isfile(vectors_schema_path):
        with open(db_schema_path, 'r') as f:
            db_json_data = json.load(f)
        with open(vectors_schema_path, 'r') as f:
            json_data = json.load(f)
        json_data.update(db_json_data)
        for table_name in json_data:
            for item in json_data[table_name]:
                constraint_type = item['constraint_type']
                column_name = item['column_name']
                foreign_table_name = item['foreign_table_name']
                foreign_column_name = item['foreign_column_name']
                rel_tuple = (table_name, constraint_type, column_name, foreign_table_name, foreign_column_name)
                rels.append(rel_tuple)
    else:
        raise ValueError(f'ERROR: Not found any schema files inside the directory {schema_dir}!')

    schema = dict()
    for entry in rels:
        if not entry[0] in schema:
            schema[entry[0]] = {'pkey': None, 'fkeys': []}
        if entry[1] == 'PRIMARY KEY':
            schema[entry[0]]['pkey'] = entry[2]
        if entry[1] == 'FOREIGN KEY':
            f_rel = (entry[2], entry[3], entry[4])
            if 'fkeys' in schema[entry[0]]:
                schema[entry[0]]['fkeys'].append(f_rel)
    return schema


def construct_relation_graph(schema, columns, blacklist):
    result = nx.MultiDiGraph()
    bl = [[y.split('.') for y in x.split('~')] for x in blacklist]
    for key in columns.keys():
        result.add_node(key, columns=[x[0] for x in columns[key]],
                        types=[x[1] for x in columns[key]],
                        pkey=schema[key]['pkey'])
    for table_name in schema.keys():
        fkeys = schema[table_name]['fkeys']
        if (len(fkeys) > 1) and (table_name not in columns):
            if table_name not in blacklist:
                relevant_fkeys = set()
                for key in fkeys:
                    if key[1] in columns.keys():
                        relevant_fkeys.add(key)
                for (key1, key2) in combinations(relevant_fkeys, 2):
                    name = table_name
                    if len(fkeys) > 2:
                        name += ':' + key1[0] + '~' + key2[0]
                    result.add_edge(
                        key1[1],
                        key2[1],
                        col1=key1[0],
                        col2=key2[0],
                        name=table_name)
        if (len(fkeys) > 0) and (table_name in columns.keys()):
            for fkey in fkeys:
                if fkey[1] in columns.keys():
                    if not fkey[0] in blacklist:
                        result.add_edge(
                            table_name, fkey[1], col1=fkey[0], col2=fkey[2], name='-')
    return result


def get_all_db_columns(columns_dir, blacklists, text_tokenization, numeric_tokenization):
    """
    Retrieves all non-unique/non-key columns, that contain text or numeric values
    text_tokenization and numeric_tokenization can be used to disable the corresponding columns
    :return dict, which assigns each table_name a list with column name, column type pairs
    """
    string_responses = []
    numeric_responses = []

    db_columns_path = columns_dir + 'db_columns.json'
    if os.path.isfile(db_columns_path):
        with open(db_columns_path, 'r') as f:
            json_data = json.load(f)
        for table_name in json_data:
            for column, type in json_data[table_name].items():
                if type in ('text', 'varchar'):
                    string_responses.append((table_name, column))
                elif type == 'integer':
                    numeric_responses.append((table_name, column))
                else:
                    raise ValueError('Column type non understood...')
    else:
        raise ValueError(f'ERROR: Not found the db_columns.json file inside the directory {columns_dir}!')

    names = defaultdict(list)
    if text_tokenization != "disabled":
        for (table, col) in string_responses:
            if (not table in blacklists[0]) and (not (table + '.' + col) in blacklists[1]):
                names[table].append((col, "string"))

    if numeric_tokenization != "disabled":
        for (table, col) in numeric_responses:
            if (not table in blacklists[0]) and (not (table + '.' + col) in blacklists[1]):
                names[table].append((col, "number"))

    return names


def main(conf):
    # Directory with db schemas
    schema_path = conf['SCHEMAS_PATH']

    # Read out data from database
    db_columns = get_all_db_columns(conf['COLUMNS_TYPE_PATH'],
        (conf['TABLE_BLACKLIST'], conf['COLUMN_BLACKLIST']), conf['TOKENIZATION_SETTINGS']['TEXT_TOKENIZATION'],
        conf['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['MODE'])

    # Construct graph from relational data
    schema = get_schema(schema_path, conf['WE_ORIGINAL_TABLE_NAME'])

    # Export schema
    relation_graph = construct_relation_graph(schema, db_columns, conf['RELATION_BLACKLIST'])

    nx.write_gml(relation_graph, conf['SCHEMA_GRAPH_PATH'])

    return
