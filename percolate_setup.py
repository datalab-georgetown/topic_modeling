import numpy as np
from nltk.corpus import stopwords
import networkx as nx
from gensim.models import Word2Vec
from scipy import spatial
from preprocessing_pipeline.NextGen import NextGen
from settings import (word_co_frequency, word_frequency,
                      normalize_frequencies, get_vocabulary)
from preprocessing_pipeline.settings import get_pp_pipeline


def get_ngrams(vocab):
    ngrams = []
    for w in list(vocab):
        if '$' in w:
            ngrams.append(w)
    return ngrams


def get_tuples_with_whitelist(tuples, whitelist):
    '''

    :param tuples:
    :param whitelist:
    :return: only the tuples that include a word in the whitelist
    '''
    wl_tuples = []
    for t1, t2 in tuples:
        for wl_word in whitelist:
            if wl_word in t1 or wl_word in t2:
                wl_tuples.append((t1, t2))
                break
    return wl_tuples


def get_tuple_counts_with_whitelist(tuples, whitelist):
    '''

    :param tuples: with cofreqeuncies
    :param whitelist:
    :return: only tuples that include a word in the whitelist,
        along with the tuple cofrequency
    '''
    wl_tuples = []
    for (t1, t2), v in tuples:
        for wl_word in whitelist:
            if wl_word in t1 or wl_word in t2:
                wl_tuples.append(((t1, t2), v))
                break
    return wl_tuples


def preprocess(dataset, extra_bigrams=[], extra_ngrams=[], blacklist_words=[],
               ngram_min_freq=256, synonyms=False, hashtags=False):
    '''

    :param dataset:
    :param extra_bigrams: if there are bigrams that you want to be included
        regardless of their frequency, add here
    :param extra_ngrams: if there are ngrams that you want to be included
        regardless of their frequency, add here
    :return processed_data: tweets fully preprocessed
    '''
    pp = get_pp_pipeline(remove_stopwords=False)
    sw = stopwords.words('english')
    sw = pp.clean_document(sw)
    sw.extend(['rt', 'twitter', 'tweet', 'dont', 'doesnt', 'know', 'like',
               'than', 'better', 'want', 'when', 'would',
               'they', 'theyr', 'wouldnt'])
    pp = get_pp_pipeline(remove_stopwords=True, stopwords=sw,
                         blacklist_words=blacklist_words,
                         synonyms=synonyms,
                         hashtags=hashtags)
    ng = NextGen()

    processed_data = ng.full_preprocess(dataset=dataset, pp=pp,
                                        extra_bigrams=extra_bigrams,
                                        extra_ngrams=extra_ngrams,
                                        ngram_min_freq=ngram_min_freq)
    return processed_data


def get_tuples(processed_data):
    '''
    Get tuples, frequencies, and cofrequencies from processed_data
    :param processed_data:
    :return word pairs that include at least one ngram, and their cofrequency:
    '''
    freqs = {}
    freqs = word_frequency(freqs, processed_data)
    nf = normalize_frequencies(freqs, len(processed_data))
    cofreqs = {}
    cofreqs = word_co_frequency(cofreqs, processed_data)
    ncf = normalize_frequencies(cofreqs, len(processed_data))
    vocab = get_vocabulary(processed_data)
    cof_tups = sorted([(x, cofreqs[x]) for x in list(cofreqs.keys())],
                      key=lambda x: x[1])
    # cleaned_tups = clean_tups(cof_tups)
    return cof_tups, nf, ncf, vocab


def get_ngram_tuples(cleaned_tups, vocab):
    '''

    :param cleaned_tups: a list of token pairs (tuples) and their cofrequencies
    :return only the tuples that include an ngram:
    '''
    ngrams = get_ngrams(vocab)
    ngram_tuples = get_tuple_counts_with_whitelist(cleaned_tups, ngrams)
    return ngram_tuples


def get_graph(cofrequency_tuples, k=0):
    '''

    :param cofrequency_tuples, ((u, v), cofrequency):
    :return G:
    '''
    G = nx.Graph()
    for e, w in cofrequency_tuples:
        if w > k and e[0] != e[1]:
            G.add_edge(e[0], e[1], weight=w)
    return G


def graph_density(G):
    return nx.density(G)


def subgraph_density(topic, G):
    sub_g = G.subgraph(topic)
    return nx.density(sub_g)


def train_word2vec_model(dataset, dataset_name, min_count=50):
    w2v = Word2Vec(dataset, min_count=min_count)
    w2v.wv.save_word2vec_format('local_{}_w2v.txt'.format(dataset_name),
                                binary=False)


def load_word2vec_model(path):
    embeddings_dict = {}
    with open('{}'.format(path), 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            if len(vector) == 100:
                embeddings_dict[word] = vector
    return embeddings_dict


def find_closest_embeddings(embedding_vector, embeddings_dict):
    return sorted(embeddings_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(
                      embeddings_dict[word], embedding_vector))


def cosine_distance(embeddings_dict, w1, w2):
    return spatial.distance.euclidean(embeddings_dict[w1], embeddings_dict[w2])
