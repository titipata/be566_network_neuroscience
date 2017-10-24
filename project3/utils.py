import numpy as np
import scipy.sparse as sp
from itertools import combinations
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import pairwise_distances
import fastcluster
import hcluster


def calculate_entropy(p, alpha):
    """
    Generalized Jensen-Shannon Divergence
    ref: https://arxiv.org/abs/1706.08671
    """
    if not sp.issparse(p):
        p = sp.csr_matrix(p.ravel())
    else:
        if alpha == 1:
            H = - np.sum(p.data * np.log(p.data))
        elif alpha == 2:
            H = 1 - (p.data ** 2).sum()
        else:
            H = ((p.data ** alpha).sum() - 1)/ (1 - alpha)
    return H


def distance_language(p1, p2, alpha=2):
    """
    Distance between two articles using Jensen-Shannon Divergence
    with given alpha
    """
    h1 = calculate_entropy(p1, alpha)
    h2 = calculate_entropy(p2, alpha)
    h12 = calculate_entropy((p1 + p2) / 2, alpha)
    d_lang = ((2 * h12) - h1 - h2)/ (0.5 * (2 - h1 - h2))
    return d_lang


def calculate_entropy_dist(abstracts, sparse=True):
    """
    Calculate entropy distance matrix between documents
    """
    count_vec_model = CountVectorizer(max_df=0.8, min_df=4,
                                      stop_words='english',
                                      lowercase=True)
    X = count_vec_model.fit_transform(abstracts)
    n_samples, n_features = X.shape
    gf = np.ravel(X.sum(axis=0))
    P = (X * sp.spdiags(1./gf, diags=0, m=n_features, n=n_features))
    D_lang = pairwise_distances(P, metric=distance_language)
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
