from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
from .pipeline import Preprocess
from .remove_punctuation import RemovePunctuation
from .remove_stopwords import RemoveStopWords
from .remove_urls import RemoveUrls
from .synonyms import Synonyms
from .stem import Stem
from .short_word import RemoveShortWords
from .capitalization import Capitalization
from .twitter_cleaner import TwitterCleaner
from .blacklist import Blacklist


def get_pp_pipeline(remove_stopwords=False, stem=True, stopwords=None,
                    blacklist_words=None, synonyms=False, hashtags=False):
    pp = Preprocess()

    rp = RemovePunctuation(keep_hashtags=hashtags)
    ru = RemoveUrls()
    cap = Capitalization()
    short_words = RemoveShortWords()
    tc = TwitterCleaner()
    # lem = Lemmatize()

    # pp.document_methods = []
    pp.document_methods = [(tc.remove_deleted_tweets, str(tc),),
                           (tc.remove_users, str(tc),),
                           (ru.remove_urls, str(ru),),
                           (rp.remove_punctuation, str(rp),),
                           (cap.lowercase, str(cap),),
    ]

    if blacklist_words:
        bl = Blacklist(blacklist_words)
        pp.document_methods.append((bl.remove_blacklist_words, str(bl)))

    if remove_stopwords:
        rsw = RemoveStopWords(extra_sw=stopwords)
        pp.document_methods.append((rsw.remove_stopwords, str(rsw)))

    if synonyms:
        syn = Synonyms()
        pp.document_methods.append((syn.replace_synonyms, str(syn)))

    if stem:
        stem = Stem()
        pp.document_methods.append((stem.stem_document, str(stem,), ))

    pp.document_methods.append((short_words.remove_short_words,
                                str(short_words), ))

    return pp


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


def or_list(booleans):
    return True in booleans


def get_frequent_ngrams(text, n, stopword_list, threshold):
    bigrams = ngrams(text, n)
    bigram_freq = Counter(bigrams)
    frequent_bigrams = []
    for bigram, freq in bigram_freq.most_common():
        if not (or_list([i in stopword_list for i in bigram])):
            if freq > threshold:
                frequent_bigrams.append('{}${}'.format(bigram[0], bigram[1]))
            else:
                break
    return frequent_bigrams


def ngrammize_text(text, ngrams):
    bigrammized_text = []
    i = 0
    while i < len(text):
        term = text[i]
        if i == len(text) - 1:
            bigrammized_text.append(term)
        else:
            next_term = text[i + 1]
            test_bigram = '{}${}'.format(term, next_term)
            if test_bigram in ngrams:
                bigrammized_text.append(test_bigram)
                i += 1
            else:
                bigrammized_text.append(term)
        i += 1
    return bigrammized_text


def get_dataset_ngrams(docs, min_freq=1000, sw=None, extra_bigrams=None,
                       extra_ngrams=None):
    if not sw:
        sw = stopwords.words('english')
        sw_pp = get_pp_pipeline(remove_stopwords=False)
        sw = sw_pp.clean_document(sw)
    full_text = []
    for doc in docs:
        full_text.extend(doc)
    frequent_bigrams = get_frequent_ngrams(full_text, 2, sw, min_freq)
    if extra_bigrams:
        frequent_bigrams.extend(extra_bigrams)
    bigrammized_text = ngrammize_text(full_text, frequent_bigrams)
    frequent_ngrams = get_frequent_ngrams(bigrammized_text, 2, sw, min_freq)
    if extra_ngrams:
        frequent_ngrams.extend(extra_ngrams)
    return frequent_bigrams, frequent_ngrams


def insert_ngrams_flat_from_lists(docs, frequent_bigrams, frequent_ngrams):
    for i in range(0, len(docs)):
        doc = docs[i]
        doc = ngrammize_text(doc, frequent_bigrams)
        doc = ngrammize_text(doc, frequent_ngrams)
        docs[i] = doc
    return docs


def insert_ngrams_flat(docs, min_freq=1000, sw=None, extra_bigrams=None,
                       extra_ngrams=None):
    fb, fn = get_dataset_ngrams(docs, min_freq, sw, extra_bigrams,
                                extra_ngrams)
    return insert_ngrams_flat_from_lists(docs, fb, fn)


def insert_ngrams_from_lists(date_doc_tuples, frequent_bigrams,
                             frequent_ngrams):
    for i in range(0, len(date_doc_tuples)):
        date, doc = date_doc_tuples[i]
        doc = ngrammize_text(doc, frequent_bigrams)
        doc = ngrammize_text(doc, frequent_ngrams)
        date_doc_tuples[i] = (date, doc)
    return date_doc_tuples


def insert_ngrams(date_docs, min_freq=1000, sw=None, extra_bigrams=None,
                  extra_ngrams=None):
    fb, fn = get_dataset_ngrams([x[1] for x in date_docs], min_freq, sw,
                                extra_bigrams, extra_ngrams)
    return insert_ngrams_from_lists(date_docs, fb, fn)
