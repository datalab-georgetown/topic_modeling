

def get_vocabulary(docs):
    '''
    This version of get_vocabulary takes 0.08 seconds on 100,000 documents
    whereas the old version took forever.
    '''
    vocab = []
    for i in range(0, len(docs)):
        vocab.extend(docs[i])
    return list(set(vocab))


def word_frequency(frequency, docs):
    '''
    :param frequency: passed explicitly so that you can increment existing
        frequencies if using in online mode
    :param docs:
    :return: updated frequency

    '''
    for doc in docs:
        for word in doc:
            if word in frequency:
                frequency[word] += 1
            else:
                frequency[word] = 1
    return frequency


def word_co_frequency(frequency, docs):
    for doc in docs:
        for i in range(0, len(doc) - 1):
            w1 = doc[i]
            for j in range(i + 1, len(doc)):
                w2 = doc[j]
                word_list = sorted([w1, w2])
                word_tup = tuple(word_list)
                if word_tup not in frequency:
                    frequency[word_tup] = 0
                frequency[word_tup] += 1
    return frequency


def normalize_frequencies(frequencies, k):
    nf = {}
    for key in frequencies.keys():
        nf[key] = frequencies[key] / k
    return nf


def load_topics(path):
    topics = []
    with open(path, 'r') as f:
        for line in f:
            words = line.strip().split(',')
            topics.append(words)
    return topics


def load_dataset(path):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            d = line.strip().split()
            dataset.append(d)
    return dataset
