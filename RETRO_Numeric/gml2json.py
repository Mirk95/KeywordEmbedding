#!/usr/bin/python3

import networkx as nx
import simplejson as json
from networkx.readwrite import json_graph


def main(conf):
    # parse the gml file and build the graph object
    g = nx.read_gml(conf['SCHEMA_GRAPH_PATH'])
    # create a dictionary in a node-link format that is suitable for JSON serialization
    d = json_graph.node_link_data(g)
    with open(conf['SCHEMA_JSON_GRAPH_PATH'], 'w') as fp:
        json.dump(d, fp, indent=4)
