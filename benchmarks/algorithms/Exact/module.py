from sklearn.neighbors import NearestNeighbors
import numpy as np
import logging

''' Note: sklearn.NearestNeighbors
sklearn.NearestNeighbors is an unsupervised learner for implementing neighbor searches.

Exact method and metric to search for the nearest neighbor could be chosen, among others parameters.

- Algorithm used to compute the nearest neighbors:
    ‘ball_tree’ will use KDTree (same function as sklearn.neighbors.KDTree)
    ‘kd_tree’ will use KDTree (same function as sklearn.neighbors.KDTree)
    ‘brute’ will use a brute-force search.
    ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

- Metrics able to use (depending on the algorithm chosen):
    (KDTree.valid_metrics): 'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity'
    (KDTree.valid_metrics):'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity', 'seuclidean', 'mahalanobis', 'hamming', 'canberra', 'braycurtis', 'jaccard', 'dice', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'haversine', 'pyfunc'
    (NearestNeighbors.VALID_METRICS['brute']): 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'haversine', 'jaccard', 'l1', 'l2', 'mahalanobis', 'manhattan', 'minkowski', 'nan_euclidean', 'precomputed', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
    
    By default, the algorithm would be chosen according to the distance metric that is wanted to be used
    (KDTree and KDTree would be preferable (only) according to its performance),
    but it can also be provided by the user on config files
'''
# Using sklearn.neighbors.nearestNeighbors, build the index of nearest neighbors using the distance metric chosen
# By default, the exact algorithm chosen to build the index that will depend on the metric
# but can be also provided by the user
def Exact_nn_index(train_set, metric, exact_algorithm):

    if exact_algorithm == 'auto':
    # Based on the metric that is going to be used, choose an exact algorithm that supports it
        if metric == 'cosine':
            exact_algorithm = 'brute'
        else:
            exact_algorithm = 'kd_tree'

    # Determine the knn of each element on the train_set and build the index accordingly (build nn estimator)
    tree_index = NearestNeighbors(metric=metric, algorithm=exact_algorithm).fit(train_set)
    logging.info("Building index using " + metric + " metric and " + exact_algorithm + " algorithm")

    return tree_index


# Given an index previously built with an exact method, find the k nearest neighbors of the elements constituting the test set
def Exact_nn_search(train_set, test_set, k, tree_index):

    # Find the knn of the test_set elements between those contained on the train_set index
    dists, indices = tree_index.kneighbors(test_set, k)

    # Get the coordinates of the found neighbors
    coords = np.array(train_set[indices])

    # Return knn and their distances with the query points
    #logging.info(str(k) + "-Nearest Neighbors found using an Exact Method + " + distance_type + " distance + " + algorithm + " algorithm.")
    print(f"Los vecinos exactos son: {indices} con distancias {dists}")
    return indices, coords, dists
