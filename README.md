This repo is based on [GraphSage repo](https://github.com/hacertilbec/GraphSAGE), a newer version of the original [GraphSAGE repo](https://github.com/williamleif/GraphSAGE/), updated for modern Python versions and packages. It includes scripts for batch jobs, enabling the execution of GraphSAGE on the PLGrid GPU cluster. Additionally, we have improved the repository's flexibility by adding scripts to prepare data in the appropriate format for GraphSAGE. A universal evaluation script for unsupervised training has also been included. We have also gathered statistics from training and evaluation. Visualizations of the results can be found in the `statistics` directory.

## GraphSage: Representation Learning on Large Graphs

#### Authors: [William L. Hamilton](http://stanford.edu/~wleif) (wleif@stanford.edu), [Rex Ying](http://joy-of-thinking.weebly.com/) (rexying@stanford.edu)
#### [Project Website](http://snap.stanford.edu/graphsage/)

#### [Alternative reference PyTorch implementation](https://github.com/williamleif/graphsage-simple/)

### Overview

This directory contains code necessary to run the GraphSage algorithm.
GraphSage can be viewed as a stochastic generalization of graph convolutions, and it is especially useful for massive, dynamic graphs that contain rich feature information.
See our [paper](https://arxiv.org/pdf/1706.02216.pdf) for details on the algorithm.

*Note:* GraphSage now also has better support for training on smaller, static graphs and graphs that don't have node features.
The original algorithm and paper are focused on the task of inductive generalization (i.e., generating embeddings for nodes that were not present during training),
but many benchmarks/tasks use simple static graphs that do not necessarily have features.
To support this use case, GraphSage now includes optional "identity features" that can be used with or without other node attributes.
Including identity features will increase the runtime, but also potentially increase performance (at the usual risk of overfitting).
See the section on "Running the code" below.

*Note:* GraphSage is intended for use on large graphs (>100,000) nodes. The overhead of subsampling will start to outweigh its benefits on smaller graphs. 

The repository contains three data directories with data in format reqired by GraphSAGE. These directories were create based on the following datasets:

- [Cora Dataset](https://github.com/williamleif/graphsage-simple/tree/master/cora)
- [Facebook Large Page-Page Network](http://snap.stanford.edu/data/facebook-large-page-page-network.html)
- [Amazon product co-purchasing network](https://snap.stanford.edu/data/com-Amazon.html)

If you make use of this code or the GraphSage algorithm in your work, please cite the following paper:

     @inproceedings{hamilton2017inductive,
	     author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
	     title = {Inductive Representation Learning on Large Graphs},
	     booktitle = {NIPS},
	     year = {2017}
	   }

### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn, and networkx are required (but networkx must be <=1.11). You can install all the required packages using the following command:

	$ pip install -r requirements.txt


### Running the code

The unsupervised.sh and supervised.sh files contain example batch jobs that enable running the unsupervised and supervised variants of GraphSage on the PLGrid GPU cluster. These scripts require passing appropriate parameters. 

For unsupervised.sh:
- train_prefix,
- model,
- identity_dim,
- max_total_steps,
- validate_iter

Example usage:

```bash
sbatch unsupervised.sh ./facebook_data/facebook graphsage_mean 128 1000 10
```

For supervised.sh:
- train_prefix,
- model,
- identity_dim,
- sigmoid_flag,


Example usage:

```bash
sbatch supervised.sh ./cora_data/cora graphsage_mean 128 false
```

If your benchmark/task does not require generalizing to unseen data, we recommend you try setting the "--identity_dim" flag to a value in the range [64,256].
This flag will make the model embed unique node ids as attributes, which will increase the runtime and number of parameters but also potentially increase the performance.
Note that you should set this flag and *not* try to pass dense one-hot vectors as features (due to sparsity).
The "dimension" of identity features specifies how many parameters there are per node in the sparse identity-feature lookup table.


*Note:* For any multi-ouput dataset that allows individual nodes to belong to multiple classes, it is necessary to set the `--sigmoid` flag during supervised training. By default the model assumes that the dataset is in the "one-hot" categorical setting.


#### Input format
As input, at minimum the code requires that a --train_prefix option is specified which specifies the following data files:

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy [optional] --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
* <train_prefix>-walks.txt [optional] --- A text file specifying random walk co-occurrences (one pair per line) (*only for unsupervised version of graphsage)

To run the model on a new dataset, you need to make data files in the format described above.
To run random walks for the unsupervised model and to generate the <prefix>-walks.txt file)
you can use the `run_walks` function in `graphsage.utils`.

#### Model variants
The user must also specify a --model, the variants of which are described in detail in the paper:
* graphsage_mean -- GraphSage with mean-based aggregator
* graphsage_seq -- GraphSage with LSTM-based aggregator
* graphsage_maxpool -- GraphSage with max-pooling aggregator (as described in the NIPS 2017 paper)
* graphsage_meanpool -- GraphSage with mean-pooling aggregator (a variant of the pooling aggregator, where the element-wie mean replaces the element-wise max).
* gcn -- GraphSage with GCN-based aggregator
* n2v -- an implementation of [DeepWalk](https://arxiv.org/abs/1403.6652) (called n2v for short in the code.)

#### Logging directory
Finally, a --base_log_dir should be specified (it defaults to the current directory).
The output of the model and log files will be stored in a subdirectory of the base_log_dir.
The path to the logged data will be of the form `<sup/unsup>-<data_prefix>/graphsage-<model_description>/`.
The supervised model will output F1 scores, while the unsupervised model will train embeddings and store them.
The unsupervised embeddings will be stored in a numpy formated file named val.npy with val.txt specifying the order of embeddings as a per-line list of node ids.
Note that the full log outputs and stored embeddings can be 5-10Gb in size (on the full data when running with the unsupervised variant).

#### Using the output of the unsupervised models

The unsupervised variants of GraphSage will output embeddings to the logging directory as described above.
These embeddings can then be used in downstream machine learning applications.
The `eval_scripts` directory contains examples of feeding the embeddings into simple logistic classifiers.

#### Acknowledgements

The original version of this code base was originally forked from https://github.com/tkipf/gcn/, and we owe many thanks to Thomas Kipf for making his code available.
We also thank Yuanfang Li and Xin Li who contributed to a course project that was based on this work.
Please see the [paper](https://arxiv.org/pdf/1706.02216.pdf) for funding details and additional (non-code related) acknowledgements.
