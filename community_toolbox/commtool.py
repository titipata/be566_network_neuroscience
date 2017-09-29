import numpy as np
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
    """
    partitions = np.array(partitions).ravel()
    locations = np.array(locations)

    comm_ave_pairwise_spatial_dist = []
    comm_ave_pairwise_spatial_dist.append((0, pairwise_distances(locations).mean()))
    for partition in np.unique(partitions):
        comm_avg = pairwise_distances(locations[np.where(partitions == partition)], metric='euclidean').mean()
        comm_ave_pairwise_spatial_dist.append((partition, comm_avg))
    return np.array(comm_ave_pairwise_spatial_dist)


def comm_laterality(partitions, categories):
    """
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
        comm_laterality_array --- an (M+1)x2 array where M is the number
            of communities.  The first column contains the community index and
            the second column contains the laterality of that community.
            The M+1th entry contains the  laterality of the partition of the
            network (denoted as community 0).
    """



    return comm_laterality_array
