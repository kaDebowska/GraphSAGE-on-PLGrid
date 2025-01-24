import os
import json
import random
import numpy as np
import networkx as nx
import pandas as pd
import argparse
import time


def prepare_graph_data(graph_file, class_file, output_dir, output_prefix, feature_file=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.Graph()

    with open(graph_file, 'rt') as f:
        first_line = f.readline().strip()
        if ',' in first_line:
            df_graph = pd.read_csv(graph_file)
            edges = df_graph.values
        else:
            edges = []
            for line in open(graph_file, 'r'):
                if line.startswith('#') or line.strip() == first_line:
                    continue
                edges.append(tuple(map(int, line.strip().split())))

        for i, j in edges:
            G.add_edge(i, j)

    df_classes = pd.read_csv(class_file)
    communities = {}
    for _, row in df_classes.iterrows():
        node_id = row['id']
        class_label = row['page_type']
        if node_id not in communities:
            communities[node_id] = []
        communities[node_id].append(class_label)

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

    def convert_to_builtin_types(data):
        if isinstance(data, dict):
            return {convert_to_builtin_types(k): convert_to_builtin_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_to_builtin_types(v) for v in data]
        elif isinstance(data, np.int64):
            return int(data)
        elif isinstance(data, np.float64):
            return float(data)
        else:
            return data

    graph_data = convert_to_builtin_types(graph_data)

    print("Nodes number: ", len(nodes))
    with open(os.path.join(output_dir, f'{output_prefix}-G.json'), 'w') as f:
        json.dump(graph_data, f)

    id_map = {int(node): idx for idx, node in enumerate(G.nodes)}
    with open(os.path.join(output_dir, f'{output_prefix}-id_map.json'), 'w') as f:
        json.dump(id_map, f)

    if feature_file:
        with open(feature_file, 'r') as f:
            raw_features = json.load(f)
        
        num_nodes = len(G.nodes)
        feature_dim = max(len(features) for features in raw_features.values())
        feats = np.zeros((num_nodes, feature_dim), dtype=int)

        for node, features in raw_features.items():
            if int(node) in G.nodes:
                feats[id_map[int(node)], :len(features)] = features

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

    with open(os.path.join(output_dir, f'{output_prefix}-class_map.json'), 'w') as f:
        json.dump(one_hot_communities, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for GraphSAGE.')
    parser.add_argument('graph_file', type=str, help='Path to the graph file.')
    parser.add_argument('class_file', type=str, help='Path to the class_file file.')
    parser.add_argument('feature_file', type=str, help='Path to the feature file.')
    parser.add_argument('output_dir', type=str, help='Path to the output folder.')
    parser.add_argument('output_prefix', type=str, help='Output prefix for the generated files.')
    

    args = parser.parse_args()
    start = time.time()
    prepare_graph_data(args.graph_file, args.class_file, args.output_dir, args.output_prefix, args.feature_file)
    end = time.time()
    elapsed_time = end - start
    print(f"Data preparation time: {elapsed_time}")