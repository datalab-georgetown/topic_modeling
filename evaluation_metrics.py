import math
import numpy as np


def split_ngrams(topics):
    split_topics = []
    for topic in topics:
        split_topic = []
        for x in topic:
            line = x.split('$')
            line.append(x)
            split_topic.extend(line)
        split_topic = list(set(split_topic))
        split_topics.append(split_topic)
    return split_topics


def words_in_topic(test, gt, topn=10):
    words = 0
    for word in test[:topn]:
        if word in gt:
            words += 1
    return words


def npmi(topic, frequencies, cofrequencies):
    v = 0
    x = len(topic)
    for i in range(0, len(topic)):
        w_i = topic[i]
        p_i = 0
        if w_i in frequencies:
            p_i = frequencies[w_i]
        for j in range(i + 1, len(topic)):
            w_j = topic[j]
            p_j = 0
            if w_j in frequencies:
                p_j = frequencies[w_j]
            word_tup = tuple(sorted([w_i, w_j]))
            p_ij = 0
            if word_tup in cofrequencies:
                p_ij = cofrequencies[word_tup]
            if p_ij == 0:
                v -= 1
            else:
                pmi = math.log(p_ij / (p_i * p_j), 2)
                denominator = -1 * math.log(p_ij, 2)
                v += (pmi / denominator)
    return (2 * v) / (x * (x - 1))


def topic_npmis(T, frequencies, cofrequencies, k=20):
    npmis = []
    for topic in T:
        n = npmi(topic[:k], frequencies, cofrequencies)
        npmis.append(n)
    return npmis


def topic_average_npmi(T, frequencies, cofrequencies, k=20):
    npmis = topic_npmis(T, frequencies, cofrequencies, k=k)
    return np.mean(npmis)


def topic_diversity(T, k):
    '''
    fraction of words in top-k words of each topic that are unique
    :param T:
    :param k: top k words per topic
    :return:
    '''
    top_words = []
    for topic in T:
        top_words.extend(topic[:k])
    unique_words = set(top_words)
    return len(unique_words) / len(top_words)
