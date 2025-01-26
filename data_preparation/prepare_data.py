import networkx as nx
import numpy as np
import gzip
import json
import random
import argparse
import os
import time

def prepare_graph_data(graph_file, output_dir, output_prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.Graph()

    with gzip.open(graph_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            i, j = map(int, line.strip().split())
            G.add_edge(i, j)

    nodes = list(G.nodes())
    random.shuffle(nodes)  
    num_val = int(0.1 * len(nodes)) 
    num_test = int(0.1 * len(nodes))  

    for i, node in enumerate(nodes):
        G.nodes[node]['val'] = False
        G.nodes[node]['test'] = False
        if i < num_val:
            G.nodes[node]['val'] = True
        elif i < num_val + num_test:
            G.nodes[node]['test'] = True

    graph_data = nx.node_link_data(G)

    print("Nodes number: ", {len(nodes)})
    with open(os.path.join(output_dir, f'{output_prefix}-G.json'), 'w') as f:
        json.dump(graph_data, f)

    id_map = {node: idx for idx, node in enumerate(G.nodes)}
    with open(os.path.join(output_dir, f'{output_prefix}-id_map.json'), 'w') as f:
        json.dump(id_map, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for GraphSAGE.')
    parser.add_argument('graph_file', type=str, help='Path to the graph file (gzip compressed).')
    parser.add_argument('output_dir', type=str, help='Path to the output folder.')
    parser.add_argument('output_prefix', type=str, help='Output prefix for the generated files.')


    args = parser.parse_args()
    start = time.time()
    prepare_graph_data(args.graph_file, args.output_dir, args.output_prefix)
    end = time.time()
    elapsed_time = end - start
    print(f"Data preparation time: {elapsed_time}")
