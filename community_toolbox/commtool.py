import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances


def comm_ave_pairwise_spatial_dist(partitions, locations):
    """
    Community Average Pairwise Spatial Distance

    Source: comm_ave_pairwise_spatial_dist.m
    Reference: Feldt Muldoon S, Soltesz I, Cossart R (2013) Spatially clustered
        neuronal assemblies comprise the microstructure of synchrony in chronically
        epileptic networks. Proc Natl Acad Sci USA 110:3567?3572.

    Inputs
    ======
    partitions: list or array of len N, N is number of nodes in network
        Each entry contains the community index for
        node i (where communities are sequentially indexed as 1:M and M
        is the total number of detected communities).
    locations: list or array of size N x dims,

    Outputs
    =======
    comm_ave_pairwise_spatial_dist: (M+1) x 2 np.array where M is the number
        of communities. The first column contains the community index and
        the second column contains the average pairwise spatial distance
        within that community.  The M+1th entry contains the average pairwise
        spatial distance of the network (denoted as community 0).

    Example
    =======
    >> partitions = [1, 2, 1, 1, 2, 3, 3]
    >> locations = np.array([[0, 1], [3,0], [0, 0],
                             [0, 2], [5, 0], [-1, -1],
                             [-1, -2]])
    >> comm_ave_pairwise_spatial_dist(partitions, locations)

    """
    partitions = np.array(partitions).ravel()
    locations = np.array(locations)

    comm_ave_pairwise_spatial_dist = []
    d = pairwise_distances(locations, metric='euclidean')
    comm_avg = d[np.triu_indices(len(d), k=1)].mean() # mean of lower diag
    # alternatively
    # np.fill_diagonal(d, np.nan)
    # comm_avg = np.nanmean(d)
    comm_ave_pairwise_spatial_dist.append((0, comm_avg))
    for partition in np.unique(partitions):
        if len(np.where(partitions == partition)[0]):
            d = pairwise_distances(locations[np.where(partitions == partition)], metric='euclidean')
            comm_avg = d[np.triu_indices(len(d), k=1)].mean()
            comm_ave_pairwise_spatial_dist.append((partition, comm_avg))
        else:
            comm_ave_pairwise_spatial_dist.append((partition, 0))
    return np.array(comm_ave_pairwise_spatial_dist)


def comm_laterality(partitions, categories):
    """
    Function to calculate the laterality of detected communities

    Inputs
    ======
        partitions: an Nx1 array where N is the number of nodes
            in the network.  Each entry contains the communitiy index for
            node i (where communities are sequentially indexed as 1:M and M
            is the total number of detected communities).

        categories: a Nx1 array where N is the number of nodes and
            each entry is either a '0' or '1' denoting the assignment of each
            node to one of two communities.

    Outputs
    =======
        comm_laterality_array: an (M+1)x2 array where M is the number
            of communities.  The first column contains the community index and
            the second column contains the laterality of that community.
            The M+1th entry contains the  laterality of the partition of the
            network (denoted as community 0).
    """

    partitions = np.array(partitions).ravel()
    categories = np.array(categories).ravel()
    categories_list = list(set(categories))

    n_nodes = len(partitions)
    n_communities = len(np.unique(partitions))
    n_surrogates = 1000

    comm_laterality = []
    number_nodes_in_communities = []
    for partition in np.unique(partitions):
        categories_i = categories[np.where(partitions == partition)]
        n_nodes_in_i = len(categories_i)
        categories_count = Counter(categories_i)
        n_0 = categories_count.get(categories_list[0], 0)
        n_1 = categories_count.get(categories_list[1], 0)
        laterality = np.abs(n_0 - n_1)/n_nodes_in_i
        number_nodes_in_communities.append((n_nodes_in_i, n_0, n_1))
        comm_laterality.append((partition, laterality))
    comm_laterality = np.array(comm_laterality)
    number_nodes_in_communities = np.array(number_nodes_in_communities)

    for _ in range(n_surrogates):
        rand_perm_assignment = np.random.permutation(range(n_nodes))


    return comm_laterality_array


def sig_perm_test(C, A, T):
    """
    Uses the permutation test to calculate
    the significance of clusters in a given community structure

    Inputs
    ======
    A: a N-by-N weighted adjacency matrix
    C: a N-by-1 partition(cluster) vector
    T: # of random permutations

    Outputs
    =======
    sig_array: the significance of all clusters
    Q: the modularity of the given partition(cluster)
    Q_r: the modularities of all random partitions
    """


    return None


def
    """
    This a function that uses the Lumped Markov chain to calculate
    the significance of clusters in a given community structure.
    refer to "Piccardi 2011 in PloS one".

    Here we normalize the original definition of persistence by
    the size of the corresponding cluster

    Inputs
    ======
        A: a N-by-N weighted adjacency matrix
        C: a N-by-1 partition(cluster) vector

    Outputs
    =======
        persistence: normalized persistence probability of all clusters

    """
    n_nodes = len(A)
    n_communities = len(np.unique(C))

    P = np.linalg.solve(np.diag(A.sum(axis=1)), A)
    eig_val, eig_vec = np.linalg.eig(P.T)

    Pi = eig_vec.T

    return persistence


def integration(MA, system_by_node):
    """
    Integration coefficient calculates the integration coefficient
    for each node of the network. The integration coefficient of a region
    corresponds to the average probability that this region is in the same
    network community as regions from other systems.

    Inputs
    ======
        MA, Module Allegiance matrix, where element (i,j)
            represents the probability that nodes i and j
            belong to the same community
        system_by_node, np.array array containing the system
            assignment for each node

    Outputs
    =======
        I, integration coefficient for each node (M x 2 array)

    Marcelo G Mattar (08/21/2014)
    """

    I = []
    MA = np.array(MA)
    system_by_node = np.array(system_by_node).ravel()
    np.fill_diagonal(MA, np.nan)

    for s in np.unique(system_by_node):
        idx = np.where(system_by_node != s)[0]
        int_coef = np.nanmean(MA[idx[:, None], idx])
        I.append((s, int_coef))

    return np.array(I)
