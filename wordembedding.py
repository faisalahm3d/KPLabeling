from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing
import os

sentences = brown.sents()
print(sentences[:4])
w2v =Word2Vec(sentences, size=10, window=5, min_count=5, negative=5, iter=20, workers=multiprocessing.cpu_count())
word_vector = w2v.wv
result = word_vector.similar_by_word("evidence")
print(result)

file = 'word2vector2.txt'
w2v.wv.save_word2vec_format(file,binary=False)

