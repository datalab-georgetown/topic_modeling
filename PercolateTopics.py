from percolate_model import (boost, greedy_density, greedy_silhouette,
                             bound_topics_by_silhouette, augment_topics,
                             prune_big_topics_density, prune_big_topics_npmi)
from percolate_setup import (preprocess, get_ngram_tuples, get_graph,
                             get_tuples, load_word2vec_model)


class PercolateTopics:
    def __init__(self, dataset=None, k=0, increment=1, c=1, ngram_min_freq=256,
                 preprocessed=False, ngram_ablation=False,
                 synonym_ablation=False, hashtag_ablation=False):
        self.k = k
        self.c = c
        self.increment = increment
        self.ngram_ablation = ngram_ablation
        self.synonym_ablation = synonym_ablation

        if dataset:
            self.setup(dataset, ngram_min_freq, preprocessed, ngram_ablation,
                       synonym_ablation, hashtag_ablation)

    def setup(self, dataset, ngram_min_freq=256, preprocessed=False,
              ngram_ablation=False, synonym_ablation=False,
              hashtag_ablation=False):
        processed_data = dataset
        if not preprocessed:
            processed_data = preprocess(dataset, ngram_min_freq=ngram_min_freq,
                                        synonyms=synonym_ablation,
                                        hashtags=hashtag_ablation)
        tuples, nf, ncf, vocab = get_tuples(processed_data)
        self.freqs = nf
        self.cofreqs = ncf
        self.vocab = vocab
        if ngram_ablation:
            self.G = get_graph(tuples, self.k)
            self.ngram_ablation = True
        else:
            ngram_tuples = get_ngram_tuples(tuples, vocab)
            self.G = get_graph(ngram_tuples, self.k)
            self.ngram_ablation = False

    def set_k(self, k):
        self.k = k

    def set_increment(self, increment):
        self.increment = increment

    def run_silhouette(self, increment=-1, c=-1, q=1, augment=True,
                       embedding_model_path=None, max_embedding_distance=3,
                       embedding_threshold=0.5):
        if increment > 0:
            self.increment = increment
        if c >= 0:
            self.c = c
        if not self.G:
            raise TypeError('You must first input a data set.')
        ccs = boost(G=self.G, k=self.k, increment=self.increment)
        topics = [list(x) for x in ccs]
        T = greedy_silhouette(topics, self.cofreqs, threshold=q)
        T, removed_topics = bound_topics_by_silhouette(T, self.cofreqs, self.c)
        if augment:
            if not embedding_model_path:
                raise ValueError('You must provide a path to the pre-trained '
                                 'embedding model')
            embeddings_dict = load_word2vec_model(embedding_model_path)
            big_T = augment_topics(T, embeddings_dict,
                                   max_distance=max_embedding_distance,
                                   threshold=embedding_threshold)
            T = prune_big_topics_npmi(T, big_T, self.freqs, self.cofreqs)
        return T, removed_topics

    def run_density(self, increment=-1, c=-1, q=1, augment=True,
                    embedding_model_path=None, max_embedding_distance=3,
                    embedding_threshold=0.5):
        if increment > 0:
            self.increment = increment
        if c >= 0:
            self.c = c
        if not self.G:
            raise TypeError('You must first input a data set.')
        ccs = boost(G=self.G, k=self.k, increment=self.increment)
        topics = [list(x) for x in ccs]
        T = greedy_density(topics, self.G, threshold=q)
        if augment:
            if not embedding_model_path:
                raise ValueError('You must provide a path to the pre-trained '
                                 'embedding model')
            embeddings_dict = load_word2vec_model(embedding_model_path)
            big_T = augment_topics(topics, embeddings_dict,
                                   max_distance=max_embedding_distance,
                                   threshold=embedding_threshold)
            T = prune_big_topics_density(T, big_T, self.G)
        return T
