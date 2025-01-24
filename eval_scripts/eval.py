from __future__ import print_function
import json
import numpy as np
import time
from networkx.readwrite import json_graph
from argparse import ArgumentParser
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/eval.py data/ amazon unsup-data/graphsage_mean_small_0.000010 test
'''

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    start = time.time()
    np.random.seed(1)
    dummy = MultiOutputClassifier(DummyClassifier(strategy="stratified"))
    dummy.fit(train_embeds, train_labels)
    
    log = MultiOutputClassifier(SGDClassifier(loss="log_loss", class_weight='balanced'), n_jobs=-1)
    log.fit(train_embeds, train_labels)

    pred_labels = log.predict(test_embeds)
    avg_f1_micro = f1_score(test_labels, pred_labels, average="micro", zero_division=0)
    avg_f1_macro = f1_score(test_labels, pred_labels, average="macro", zero_division=0)
    
    print(f"Average F1 score (micro): {avg_f1_micro}")
    print(f"Average F1 score (macro): {avg_f1_macro}")

    dummy_pred_labels = dummy.predict(test_embeds)
    dummy_avg_f1_micro = f1_score(test_labels, dummy_pred_labels, average="micro", zero_division=0)
    dummy_avg_f1_macro = f1_score(test_labels, dummy_pred_labels, average="macro", zero_division=0)

    end = time.time()
    elapsed_time = end - start

    print(f"Random baseline Average F1 score (micro): {dummy_avg_f1_micro}")
    print(f"Random baseline Average F1 score (macro): {dummy_avg_f1_macro}")
    print(f"Evaluation time: {elapsed_time}")

if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation.")
    parser.add_argument("dataset_dir", help="Path to directory containing the dataset.")
    parser.add_argument("data_prefix", help="Data prefix.")
    parser.add_argument("embed_dir", help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("setting", help="Either val or test.")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    data_prefix = args.data_prefix
    data_dir = args.embed_dir
    setting = args.setting

    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(dataset_dir + f"{data_prefix}-G.json")))
    labels = json.load(open(dataset_dir + f"{data_prefix}-class_map.json"))
    labels = {int(i): l for i, l in labels.items()}
    
    train_ids = [n for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']]
    test_ids = [n for n in G.nodes() if G.nodes[n][setting]]
    train_labels = np.array([labels[i] for i in train_ids if i in labels])
    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.array([labels[i] for i in test_ids if i in labels])
    if test_labels.ndim == 1:
        test_labels = np.expand_dims(test_labels, 1)
    
    from collections import Counter
    train_class_distribution = Counter(train_labels.flatten())
    test_class_distribution = Counter(test_labels.flatten())

    print(f"Training class distribution: {train_class_distribution}")
    print(f"Test class distribution: {test_class_distribution}")

    print("running", data_dir)

    if data_dir == "feat":
        print("Using only features..")
        feats = np.load(dataset_dir + f"{data_prefix}-feats.npy")
        ## Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:,0] = np.log(feats[:,0]+1.0)
        feats[:,1] = np.log(feats[:,1]-min(np.min(feats[:,1]), -1))
        feat_id_map = json.load(open(dataset_dir + f"{data_prefix}-id_map.json"))
        feat_id_map = {int(id):val for id,val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]] 
        test_feats = feats[[feat_id_map[id] for id in test_ids]] 
        print("Running regression..")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)
        run_regression(train_feats, train_labels, test_feats, test_labels)
    else:
        embeds = np.load(data_dir + "/val.npy")
        id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[int(line.strip())] = i
        train_embeds = embeds[[id_map[id] for id in train_ids if id in id_map]] 
        test_embeds = embeds[[id_map[id] for id in test_ids if id in id_map]] 

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)
