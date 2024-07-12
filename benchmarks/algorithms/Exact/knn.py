from data.load_train_test_set import *
from benchmarks.neighbors_utils import *
from benchmarks.algorithms.Exact.module import Exact_nn_index, Exact_nn_search
import logging
from timeit import default_timer as timer


def Exact(config_file):

    # Read config file containing experiment's parameters
    dataset, k, distance, method, exact_algorithm = read_config_file(config_file)

    # Set log configuration
    logging.basicConfig(filename="./benchmarks/logs/" + dataset + "/test_knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + ".log", filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info('                          plotting Searching')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " and " + exact_algorithm
                 + " algorithm over " + str(dataset) + " dataset using " + str(distance) + " distance. ----")


    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"

    # Load the train and test sets to carry on the benchmarks
    # train_set, test_set = load_train_test(str(dataset))
    train_set, test_set = load_train_test_h5py(file_name)

    # GENERATE INDEX AND CENTROIDS
    # AND FIND THE plotting FROM THE train_set OF THE ELEMENTS CONTAINED IN THE test_set, USING DISTANCE CHOOSEN

    # Using Exact Algorithm, build the index of nearest neighbors
    start_time_i = timer()
    knn_index = Exact_nn_index(train_set, distance, exact_algorithm)
    end_time_i = timer()
    logging.info('Index time= %s seconds', end_time_i - start_time_i)

    # Store index on disk to obtain its size
    # with open("./algorithms/Exact/" + dataset + str(k) +".pickle", 'wb') as handle:
    #    dump(knn_index, handle)

    # Using Exact Algorithm and the index built, find the k nearest neighbors
    start_time_s = timer()
    indices, coords, dists = Exact_nn_search(train_set, test_set, k, knn_index)
    end_time_s = timer()
    logging.info('Search time = %s seconds\n', end_time_s - start_time_s)
    logging.info('Average time spended in searching a single point = %s',
                 (end_time_s - start_time_s) / test_set.shape[0])
    logging.info('Speed (points/s) = %s\n', test_set.shape[0] / (end_time_s - start_time_s))

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors(indices, coords, dists, file_name)

    # Print
    # print_knn(train_set, test_set, coords, dataset_name, d, "Exact", k)

    logging.info('------------------------------------------------------------------------\n')
