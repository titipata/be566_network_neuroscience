import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist, pdist
import fastcluster
import hcluster
from collections import Counter


def calculate_entropy(p, alpha=2):
    """
    Generalized Jensen-Shannon Divergence
    ref: https://arxiv.org/abs/1706.08671
    """
    return 1 - (np.sum(p) ** 2)


def distance_language(p1, p2):
    """
    Distance between two articles using Jensen-Shannon Divergence
    with alpha = 2
    """
    h1 = calculate_entropy(p1)
    h2 = calculate_entropy(p2)
    h12 = calculate_entropy((p1 + p2)/2)
    d_lang = ((2 * h12) - h1 - h2)/ (0.5 * (2 - h1 - h2))
    return d_lang


def calculate_entropy_dist(abstracts):
    """
    Calculate entropy distance matrix between documents
    """
    X = count_vec_model.fit_transform(abstracts)
    n_samples, n_features = X.shape
    gf = np.ravel(X.sum(axis=0))
    P = (X * sp.spdiags(1./gf, diags=0, m=n_features, n=n_features))
    P = P.toarray()
    d_lang = pdist(P, lambda u, v: distance_language(u, v))
    D_lang = squareform(d_lang)
    D_lang = np.nan_to_num(D_lang)
    return D_lang


def clustering(D_lang):
    """
    Hierarchical clustering on language matrix
    """
    linkage = fastcluster.linkage(D_lang_array,
                                  method='centroid',
                                  preserve_input=True)
    partition = hcluster.fcluster(linkage,
                                  t=8,
                                  criterion='distance') # distance
    gr = [k for (k, v) in  sorted(list(Counter(partition).items()),
          key=lambda x: x[1], reverse=True)]
    idx = np.concatenate([np.where(partition == g)[0] for g in gr]) # sort by group size
    return partition, idx
