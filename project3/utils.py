import numpy as np
import scipy.sparse as sp
from itertools import combinations
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import pairwise_distances
import fastcluster
import hcluster
from sklearn.preprocessing import normalize


def calculate_entropy(p, alpha):
    """
    Generalized Jensen-Shannon Divergence
    ref: https://arxiv.org/abs/1706.08671
    """
    if not sp.issparse(p):
        p = sp.csr_matrix(p.ravel())

    if alpha == 0:
        H = p.shape[1] - 1
    elif alpha == 1:
        H = - np.sum(p.data * np.log(p.data))
    elif alpha == 2:
        H = 1 - (p.data ** 2).sum()
    else:
        H = ((p.data ** alpha).sum() - 1)/ (1 - alpha)
    return H


def calculate_max_entropy(h1, h2, pi1=0.5, pi2=0.5, alpha=2):
    """
    Maximum entropy

    h1 : entropy of probability distribution p1
    h2 : entropy of probability distribution p2
    """
    if alpha == 1:
        d_max = - pi1  * np.log(pi1) - pi2 * np.log(pi2)
    else:
        d_max = (pi1 ** alpha - pi1) * h1 + \
            (pi2 ** alpha - pi2) * h2 + \
            (pi1 ** alpha + pi2 ** alpha - 1)/(1 - alpha)
    return d_max


def distance_language(p1, p2, alpha=2, norm=True):
    """
    Distance between two articles using Jensen-Shannon Divergence
    with given alpha
    """
    h1 = calculate_entropy(p1, alpha)
    h2 = calculate_entropy(p2, alpha)
    h12 = calculate_entropy(0.5 * p1 + 0.5 * p2, alpha=alpha)
    d_lang = h12 - 0.5 * h1 - 0.5 * h2
    if norm:
        d_max = calculate_max_entropy(h1, h2, pi1=0.5, pi2=0.5, alpha=alpha)
        d_lang = d_lang / d_max
    return d_lang


def calculate_entropy_dist(texts, max_df=0.8, min_df=4,
                           stop_words='english',
                           max_features=20000,
                           lowercase=True,
                           metric='jensen-shannon'):
    """
    Calculate entropy distance matrix between texts in the fields

    texts: list of texts each contains all text in the fields
    alpha: int, default to 2, you can change default value in distance_language function

    """
    count_vec_model = CountVectorizer(max_df=max_df, min_df=min_df,
                                      stop_words=stop_words,
                                      max_features=max_features,
                                      lowercase=lowercase)
    X = count_vec_model.fit_transform(texts)
    n_samples, n_features = X.shape
    P = normalize(X, axis=1, norm='l1') # get probability distribution
    if metric == 'jensen-shannon':
        D_lang = pairwise_distances(P, metric=distance_language)
    else:
        D_lang = pairwise_distances(P, metric=metric)
    return D_lang


def clustering(D_lang):
    """
    Hierarchical clustering on language matrix
    """
    linkage = fastcluster.linkage(D_lang,
                                  method='centroid',
                                  preserve_input=True)
    partition = hcluster.fcluster(linkage,
                                  t=8,
                                  criterion='distance') # distance
    gr = [k for (k, v) in  sorted(list(Counter(partition).items()),
          key=lambda x: x[1], reverse=True)]
    idx = np.concatenate([np.where(partition == g)[0] for g in gr]) # sort by group size
    return partition, idx
