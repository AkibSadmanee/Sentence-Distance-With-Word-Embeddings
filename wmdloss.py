import numpy as np
from scipy import spatial
from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_skipgram_dim300_iter35.model")

def wmdloss(s1, s2):
    return (1 - model.wmdistance(s1, s2))
