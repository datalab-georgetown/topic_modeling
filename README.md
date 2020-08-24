## Percolation-based Topic Modeling for Twitter

##### Rob Churchill and Lisa Singh
##### Version 1.0

---

This repository contains Python3 source code for the Percolation-based Topic 
Model presented in the WISDOM'20 workshop.

The Percolation-based Topic Model works by finding frequent ngrams in a data 
set, and creating a graph of co-occurrences between ngrams (n > 1) and all 
other tokens.  It then breaks down the graph into small topic kernels by 
increasing the minimum edge weight to remain in the graph.  Finally, topic 
kernels are combined, or percolated, based on a graph metric such as NPMI, 
Silhouette score, or graph density.

The repository contains the model code (`percolate_model.py`,
`percolate_setup.py`, and `PercolateTopics.py`), evaluation metrics used in 
the experiments for the paper (`evaluation_metrics.py`), and a preprocessing
pipeline used to process the data sets in our experiments
(`preprocessing_pipeline/`).

The `percolate_client.py` file contains a cookie-cutter example of how to use
the Percolation-based Topic Model.
The model expects a data set to be a list of documents, where each document is 
a list of words (i.e. `dataset = [[w1, w2, w3], [w4, w5, w6], ...]`).

--- 

###Installation

This code relies on a short list of python packages, and comes with a 
virtual environment with the packages pre-installed.  To use it, from the 
root directory, run `$ source env/bin/activate`.  If you wish to use your own
environment, run `pip install -r requirements.txt` from the root directory.

This code also relies on NLTK corpora.  To install them, run 
`$ python setup.py` once the packages have been installed or activated.

#### Word Embedding Vectors

The model, if you choose the augmented version (default), relies on a word
embedding space.

Pre-trained vectors from GloVe can be downloaded 
[here](https://nlp.stanford.edu/projects/glove/).

You can also train your own vectors using our function in `percolate_setup.py`.
```
train_word2vec_model(dataset, dataset_name, min_count=50):
```
The function uses [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 
and saves the file to `local_[dataset_name]_w2v.txt`.

#####
Our example uses a subset of the data set 
[Twenty Newsgroups](http://qwone.com/~jason/20Newsgroups/).
