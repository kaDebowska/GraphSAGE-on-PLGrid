import networkx as nx
import numpy as np
import gzip
import json
import random
import argparse
import os
import time

def prepare_graph_data(graph_file, class_file, output_dir, output_prefix, feature_file=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.Graph()

    with gzip.open(graph_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            i, j = map(int, line.strip().split())
            G.add_edge(i, j)

    # Read class_map.json to get nodes with labels
    with gzip.open(class_file, 'rt') as f:
        communities = {}
        for idx, line in enumerate(f):
            if line.startswith('#'):
                continue
            community = list(map(int, line.strip().split()))
            for node in community:
                if str(node) not in communities:
                    communities[str(node)] = []
                communities[str(node)].append(idx)

    # Filter graph to keep only nodes present in filtered_communities
    G = G.subgraph([int(node) for node in communities.keys()])

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

    if feature_file:
        feats = np.load(feature_file)
        np.save(os.path.join(output_dir, f'{output_prefix}-feats.npy'), feats)

    all_classes = sorted(list({cls for class_list in communities.values() for cls in class_list}))
    class_to_index = {cls: i for i, cls in enumerate(all_classes)}
    num_classes = len(all_classes)
    print("num_classes", num_classes)
    one_hot_communities = {}
    for node, class_list in communities.items():
        one_hot_vector = np.zeros(num_classes, dtype=int)
        for cls in class_list:
            one_hot_vector[class_to_index[cls]] = 1
        one_hot_communities[node] = one_hot_vector.tolist()

    with open(os.path.join(output_dir,f'{output_prefix}-class_map.json'), 'w') as f:
        json.dump(one_hot_communities, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for GraphSAGE.')
    parser.add_argument('graph_file', type=str, help='Path to the graph file (gzip compressed).')
    parser.add_argument('class_file', type=str, help='Path to the class_file file (gzip compressed).')
    parser.add_argument('output_dir', type=str, help='Path to the output folder.')
    parser.add_argument('output_prefix', type=str, help='Output prefix for the generated files.')
    parser.add_argument('--feature_file', type=str, help='Path to the feature file (optional).')


    args = parser.parse_args()
    start = time.time()
    prepare_graph_data(args.graph_file, args.class_file, args.output_dir, args.output_prefix, args.feature_file)
    end = time.time()
    elapsed_time = end - start
    print(f"Data preparation time: {elapsed_time}")
