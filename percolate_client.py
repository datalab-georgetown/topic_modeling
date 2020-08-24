from PercolateTopics import PercolateTopics
from sklearn.datasets import fetch_20newsgroups


raw_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers',
                                                      'quotes'),
                              categories=['sci.space'])
dataset = [d.strip().split(' ') for d in raw_data.data]

tm = PercolateTopics(dataset=dataset, k=2, increment=2, ngram_min_freq=8)
topics, _ = tm.run_silhouette(c=0, embedding_model_path=
                           'data/glove.twitter.27B.50d.txt')

print(topics)
