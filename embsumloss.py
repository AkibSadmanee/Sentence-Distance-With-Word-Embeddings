import numpy as np
from scipy import spatial
from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_skipgram_dim300_iter35.model")

def meanloss(s1, s2):
    sent1_vecs = []
    sent2_vecs = []

    for word in s1.split():
        sent1_vecs.append(model.wv[word])
        
    for word in s2.split():
        sent2_vecs.append(model.wv[word])

    sent1_vecs = np.array(sent1_vecs)
    sum1 = sent1_vecs.sum(axis=0)

    sent2_vecs = np.array(sent2_vecs)
    sum2 = sent2_vecs.sum(axis=0)
    return (1 - spatial.distance.cosine(sum1, sum2))