import math
from statistics import mean
import networkx as nx
from networkx.algorithms.connectivity import is_k_edge_connected
from percolate_setup import (cosine_distance, find_closest_embeddings,
     subgraph_density)


def boost_graph(G, k=1, weight='weight'):
    '''

    :param G:
    :param k, min frequency, only adds edges that are higher than this:
    :param weight, if your graph weights are stored in data as something other
        than 'weight', change this:
    :return g_prime, which is a new version of G with a new minimum_frequency:
    '''
    g_prime = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d[weight] > k and u != v:
            g_prime.add_edge(u, v, weight=d[weight])
    return g_prime


def gets_ccs(G):
    '''

    :param G:
    :return list of sets, where each set is a CC,
        sorted by size (greatest first):
    '''
    cc_list = list(nx.connected_components(G))
    cc_lens = [(x, len(x)) for x in cc_list]
    cc_list = [x[0] for x in sorted(cc_lens, key=lambda x: -x[1])]
    return cc_list


def boost_one_increment(G, k=0, increment=1, ccs=[], weight='weight'):
    '''
    Boost G by one increment, return g_k, new_k, all CCs
    :param G: g_k
    :param k:
    :param increment: How much to increase k by
    :param ccs: The CCs that exist at g_k and before
    :param weight: the name of the weight variable for each edge
    :return: g_k+increment, new k, new list of CCs
    '''
    new_k = k + increment
    g_k = boost_graph(G, new_k, weight=weight)
    new_ccs = gets_ccs(g_k)
    actual_new_ccs = []
    all_ccs = []
    if len(new_ccs) > 0:
        all_ccs.append(new_ccs[0])
    if len(ccs) > 1:
        all_ccs.extend(ccs[1:])
    for cc in new_ccs[1:]:
        is_subset = False
        for old_cc in all_ccs[1:]:
            if cc.issubset(old_cc):
                is_subset = True
                break
        if not is_subset:
            actual_new_ccs.append(cc)
    all_ccs.extend(actual_new_ccs)
    return g_k, new_k, all_ccs


def boost(G, k=0, increment=1, ccs=[], weight='weight'):
    '''
    Wrapper function for the entire boosting process.
    :param G: Initial G
    :param k: Initial k
    :param increment: Amount to increase k by at each step
    :param ccs: Any initial CCs you want in there... should be empty.
    :param weight: Edge weight variable
    :return: All CCs retrieved from boosting process
    '''
    g_ki = nx.Graph()
    for u, v, d in G.edges(data=True):
        g_ki.add_edge(u, v, weight=d[weight])
    gs = [g_ki]
    main_cc = g_ki
    while len(main_cc.nodes) > 1 and \
            not is_k_edge_connected(main_cc, len(main_cc.nodes) - 1):
        # print(len(g_ki.edges()))
        g_k, k, ccs = boost_one_increment(g_ki, k, increment, ccs=ccs,
                                          weight=weight)
        # print(len(ccs), len(ccs[0]))
        g_ki = g_k
        gs.append(g_ki)
        main_cc = g_ki.subgraph(ccs[0])
    return ccs


def greedy_density(topics, G, threshold=1):
    topics = sorted(topics, key=lambda x: len(x), reverse=True)
    old_len_t = 0
    while old_len_t != len(topics):
        old_len_t = len(topics)
        i = 0
        while i < len(topics) - 1:
            t_i = topics[i]
            j = i + 1
            while j < len(topics):
                t_j = topics[j]
                t_ij = list(set(t_i).union(set(t_j)))
                density_t_ij = subgraph_density(t_ij, G)
                density_t_i = subgraph_density(t_i, G)
                density_t_j = subgraph_density(t_j, G)
                base_density = 0
                if len(t_i) > len(t_j):
                    base_density = density_t_i
                elif len(t_i) < len(t_j):
                    base_density = density_t_j
                else:
                    base_density = (density_t_i + density_t_j) / 2
                if density_t_ij >= threshold * base_density:
                    # replace t_i with t_ij, remove t_j,
                    # do NOT increase j b/c size of topics has decreased by one
                    topics[i] = t_ij
                    t_i = t_ij
                    topics.remove(t_j)
                else:
                    j += 1
            i += 1
    return topics


def mean_cof(topic, token, cofrequencies):
    if len(topic) < 2:
        return 0
    cof_count = 0
    for w in topic:
        if token != w:
            word_tup = tuple(sorted([token, w]))
            if word_tup in cofrequencies:
                cof_count += cofrequencies[word_tup]
    if token in topic:
        return cof_count / (len(topic) - 1)
    return cof_count / len(topic)


def silhouette(T, topic, token, cofrequencies):
    '''
    Maximizing mean cofrequency instead of minimizing distance.
    Silhouette value of a given token from the given topic
    :param T: topic set
    :param topic: home topic of queried token
    :param token: queried token we wish to get silhouette value for
    :param cofrequencies: dictionary of cofrequencies in data set
    :return:
    '''
    a = mean_cof(topic, token, cofrequencies)
    b = 0
    for i in range(0, len(T)):
        t_i = T[i]
        if t_i != topic:
            topic_score = mean_cof(t_i, token, cofrequencies)
            if topic_score > b:
                b = topic_score
    if a == b:
        return 0
    return (a - b) / max(a, b)


def topic_silhouette(T, topic, cofrequencies):
    silhouettes = []
    for w in topic:
        silhouettes.append(silhouette(T, topic, w, cofrequencies))
    return silhouettes


def greedy_silhouette(topics, cofrequencies, threshold=1):
    topics = sorted(topics, key=lambda x: len(x), reverse=True)
    old_len_t = 0
    while old_len_t != len(topics):
        old_len_t = len(topics)
        i = 0
        while i < len(topics) - 1:
            t_i = topics[i]
            j = i + 1
            while j < len(topics):
                t_j = topics[j]
                t_ij = list(set(t_i).union(set(t_j)))
                s_t_ij = mean(topic_silhouette(topics, t_ij, cofrequencies))
                s_t_i = mean(topic_silhouette(topics, t_i, cofrequencies))
                s_t_j = mean(topic_silhouette(topics, t_j, cofrequencies))
                base_s = 0
                if len(t_i) > len(t_j):
                    base_s = s_t_i
                elif len(t_i) < len(t_j):
                    base_s = s_t_j
                else:
                    base_s = (s_t_i + s_t_j) / 2
                if s_t_ij >= max(0, threshold * base_s):
                    # replace t_i with t_ij, remove t_j,
                    # do NOT increase j b/c size of topics has decreased by one
                    topics[i] = t_ij
                    t_i = t_ij
                    topics.remove(t_j)
                else:
                    j += 1
            i += 1
    return topics


def get_silhouettes(topics, cofrequencies, normalize=True):
    silhouettes = []
    for topic in topics:
        v = mean(topic_silhouette(topics, topic, cofrequencies))
        if normalize:
            le = len(topic)
            v = v / ((le * (le - 1)) / 2)
        silhouettes.append(v)
    return silhouettes


def bound_topics_by_silhouette(topics, cofrequencies, c=1):
    topics_i = topics.copy()
    silhouettes = get_silhouettes(topics_i, cofrequencies)
    i = 0
    removed_topics = []
    while i < len(silhouettes):
        if silhouettes[i] > 1 - (c * 2) / \
                (len(topics_i[i]) * (len(topics_i[i]) - 1)):
            removed_topics.append((topics_i[i], 'high', silhouettes[i]))
            topics_i.remove(topics_i[i])
            silhouettes.remove(silhouettes[i])
        elif silhouettes[i] < (c * 2) / \
                (len(topics_i[i]) * (len(topics_i[i]) - 1)):
            removed_topics.append((topics_i[i], 'low', silhouettes[i]))
            topics_i.remove(topics_i[i])
            silhouettes.remove(silhouettes[i])
        else:
            i += 1
    return topics_i, removed_topics


def augment_topics(topics, embeddings_dict, max_distance=3, threshold=0.5):
    big_topics = []
    for t1 in topics:
        bt1 = []
        bt1.extend(t1)
        close_words = []
        for i in range(0, len(t1)):
            if t1[i] in embeddings_dict:
                closest_words = find_closest_embeddings(
                    embeddings_dict[t1[i]],
                    embeddings_dict
                )[1:max_distance + 1]
                added_words = []
                for w in closest_words:
                    wd = cosine_distance(embeddings_dict, t1[i], w)
                    if wd < threshold and w not in t1 and w not in close_words:
                        added_words.append(w)
                close_words.extend(added_words)
        bt1.extend(close_words)
        big_topics.append(bt1)
    return big_topics


def density_from_original(topic, big_topic, G):
    d = [0] * len(big_topic)
    density_original = subgraph_density(topic, G)
    for i in range(0, len(big_topic)):
        w_i = big_topic[i]
        if w_i in topic:
            d[i] = 1
            continue
        test_topic = []
        test_topic.extend(topic)
        test_topic.append(w_i)
        test_density = subgraph_density(test_topic, G)
        if test_density >= density_original:
            d[i] = 1
    final_topic = []
    for i in range(0, len(d)):
        if d[i] == 1:
            final_topic.append(big_topic[i])
    return final_topic


def prune_big_topics_density(topics, big_topics, G):
    density_topics = []
    for i in range(0, len(big_topics)):
        bt1 = big_topics[i]
        t1 = topics[i]
        density_topic = density_from_original(t1, bt1, G)
        density_topics.append(density_topic)


def avg_npmi_from_original(topic, big_topic, frequencies, cofrequencies):
    v = [0] * len(big_topic)
    for i in range(0, len(big_topic)):
        w_i = big_topic[i]
        p_i = 0
        if w_i in frequencies:
            p_i = frequencies[w_i]
        for j in range(0, len(topic)):
            w_j = topic[j]
            p_j = 0
            if w_j in frequencies:
                p_j = frequencies[w_j]
            if w_j == w_i:
                continue
            word_tup = tuple(sorted([w_i, w_j]))
            p_ij = 0
            if word_tup in cofrequencies:
                p_ij = cofrequencies[word_tup]
            if p_ij == 0:
                v[i] -= 1
            else:
                pmi = math.log(p_ij / (p_i * p_j), 2)
                denominator = -1 * math.log(p_ij, 2)
                v[i] += (pmi / denominator)
    return v


def prune_big_topics_npmi(topics, big_topics, freqs, cofreqs):
    avg_npmis = []
    for i in range(0, len(topics)):
        t1 = topics[i]
        bt1 = big_topics[i]
        avg_npmis.append(avg_npmi_from_original(t1, bt1, freqs, cofreqs))

    final_topics = []
    for i in range(0, len(big_topics)):
        t1 = topics[i]
        bt1 = big_topics[i]
        ft1 = []
        avg_npmi = avg_npmis[i]
        for j in range(0, len(bt1)):
            w_j = bt1[j]
            npmi_j = avg_npmi[j]
            if npmi_j > 0 or w_j in t1:
                ft1.append(w_j)
        final_topics.append(ft1)
    return final_topics
