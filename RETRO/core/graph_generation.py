#!/usr/bin/python3

import sys
import networkx as nx
from collections import defaultdict
from itertools import combinations
from enum import Enum

import config
import db_connection as db_con


def get_schema(con, cur):
    rel_query = ("SELECT tc.table_name, tc.constraint_type, kcu.column_name, " +
                 "ccu.table_name AS foreign_table_name, " +
                 "ccu.column_name AS foreign_column_name " +
                 "FROM information_schema.table_constraints AS tc " +
                 "JOIN information_schema.key_column_usage AS kcu " +
                 "ON tc.constraint_name = kcu.constraint_name " +
                 "AND tc.table_schema = kcu.table_schema " +
                 "JOIN information_schema.constraint_column_usage AS ccu " +
                 "ON ccu.constraint_name = tc.constraint_name " +
                 "AND ccu.table_schema = tc.table_schema")
    cur.execute(rel_query)
    rels = cur.fetchall()
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


def get_all_db_columns(blacklists, text_tokenization, numeric_tokenization, con, cur):
    """
    Retrieves all non-unique/non-key columns, that contain text or numeric values

    text_tokenization and numeric_tokenization can be used to disable the corresponding columns

    :return dict, which assigns each table_name a list with column name, column type pairs
    """

    # string values
    string_query = ("SELECT cols.table_name, cols.column_name "
                    + "FROM information_schema.columns cols "
                    + "LEFT JOIN information_schema.key_column_usage keys "
                    + "ON cols.table_name = keys.table_name AND cols.column_name = keys.column_name "
                    + "WHERE cols.table_schema = 'public' AND keys.column_name IS NULL AND cols.udt_name = 'varchar'")

    # numeric values
    numeric_query = ("SELECT cols.table_name, cols.column_name "
                     + "FROM information_schema.columns cols "
                     + "LEFT JOIN information_schema.key_column_usage keys "
                     + "ON cols.table_name = keys.table_name AND cols.column_name = keys.column_name "
                     + "WHERE cols.table_schema = 'public' AND keys.column_name IS NULL "
                     + "AND cols.udt_name IN ('smallint', 'integer', 'bigint', 'int2', 'int4', 'int8', "
                     + "'decimal', 'numeric', 'real', 'double precision', 'float4', 'float8')")

    names = defaultdict(list)
    if text_tokenization != "disabled":
        cur.execute(string_query)
        string_responses = cur.fetchall()
        for (table, col) in string_responses:
            if (not table in blacklists[0]) and (not (table + '.' + col) in blacklists[1]):
                names[table].append((col, "string"))

    if numeric_tokenization != "disabled":
        cur.execute(numeric_query)
        numeric_responses = cur.fetchall()
        for (table, col) in numeric_responses:
            if (not table in blacklists[0]) and (not (table + '.' + col) in blacklists[1]):
                names[table].append((col, "number"))

    return names


def main(argc, argv):
    # get database connection
    db_config = db_con.get_db_config(path=argv[2])
    con, cur = db_con.create_connection(db_config)

    # get retrofitting config
    conf = config.get_config(argv)

    # read out data from database
    db_columns = get_all_db_columns(
        (conf['TABLE_BLACKLIST'], conf['COLUMN_BLACKLIST']), conf['TOKENIZATION_SETTINGS']['TEXT_TOKENIZATION'],
        conf['TOKENIZATION_SETTINGS']['NUMERIC_TOKENIZATION']['MODE'], con, cur)

    # construct graph from relational data
    schema = get_schema(con, cur)

    # export schema
    relation_graph = construct_relation_graph(
        schema, db_columns, conf['RELATION_BLACKLIST'])

    nx.write_gml(relation_graph, conf['SCHEMA_GRAPH_PATH'])

    return


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
