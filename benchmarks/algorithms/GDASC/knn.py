import GDASC.gdasc_ as gdasc
from timeit import default_timer as timer
import data.load_train_test_set as lts
from benchmarks.neighbors_utils import *


def GDASC(config_file):

    # Read config file containing experiment's parameters
    dataset, k, distance, method, tam_grupo, n_centroides, radio, algorithm, implementation = read_config_file(config_file)

    # Set log configuration
    logging.basicConfig(filename="./benchmarks/logs/" + dataset + "/test_knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + "_tg" + str(tam_grupo) + "_nc" + str(n_centroides) + "_r" + str(radio) + "_" + str(algorithm) + "_" + str(implementation) + ".log", filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
    logging.info('------------------------------------------------------------------------')
    logging.info('                          plotting Searching')
    logging.info('------------------------------------------------------------------------\n')
    logging.info("")
    logging.info("---- Searching the " + str(k) + " nearest neighbors within " + method + " over " + str(
        dataset) + " dataset using " + str(distance) + " distance. ----")
    logging.info("")
    logging.info('---- GDASC Parameters - tam_grupo=%s - n_centroids=%s - radius=%s - algorithm=%s - implementation=%s ----', tam_grupo, n_centroides, radio, algorithm, implementation)

    # Regarding the dataset name, set the file name to load the train and test set
    file_name = "./data/" + str(dataset) + "_train_test_set.hdf5"


    # 1º Leemos los datos

    # Read train and test set from preprocesed h5py file
    vector_training, vector_testing = lts.load_train_test_h5py(file_name)

    # Read train and test set from original file
    # vector_training, vector_testing = lts.load_train_test(str(dataset))

    dimensionalidad = vector_testing.shape[1]
    cant_ptos = len(vector_training)


    # 2º Generamos el árbol
    n_capas, grupos_capa, puntos_capa, labels_capa = gdasc.create_tree(cant_ptos, tam_grupo, n_centroides,
                                                                     distance, vector_training, dimensionalidad, algorithm, implementation)

    # Store index in a file
    # with open("./algorithms/GDASC_MNIST" + str(k) +".pickle", 'wb') as handle:
    #    dump((puntos_capa, labels_capa), handle)


    # 3º Buscamos los k vecinos de los puntos de testing
    start_time_s = timer()

    # Creaamos las estructuras para almacenar los futuros vecinos
    indices_vecinos = np.empty([len(vector_testing), k], dtype=int)
    coords_vecinos = np.empty([len(vector_testing), k, vector_testing.shape[1]], dtype=float)
    dists_vecinos = np.empty([len(vector_testing), k], dtype=float)

    # For every point in the testing set, find its k nearest neighbors
    for i in range(len(vector_testing)):

        punto = vector_testing[i]
        #logging.info('punto %s =%s', i,  punto)

        start_time_iter = timer()

        vecinos_i = gdasc.recursive_approximate_knn_search(n_capas, n_centroides, punto, vector_training, k, distance, grupos_capa, puntos_capa, labels_capa, dimensionalidad, float(radio))
        #vecinos_i = gdasc.knn_approximate_search(n_centroides, punto, vector_training, k, distance, grupos_capa, puntos_capa, labels_capa, dimensionalidad, float(radio))

        end_time_iter = timer()
        #logging.info('Index time= %s seconds', end_time_iter - start_time_iter)
        #logging.info('punto %s - time= %s seconds', i, end_time_iter - start_time_iter)

        indices_vecinos[i] = vecinos_i[0]
        coords_vecinos[i] = vecinos_i[1]
        dists_vecinos[i] = vecinos_i[2]


    end_time_s = timer()
    logging.info('Search time = %s seconds\n', end_time_s - start_time_s)
    logging.info('Average time spent in searching a single point = %s', (end_time_s - start_time_s)/vector_testing.shape[0])
    logging.info('Speed (points/s) = %s\n', vector_testing.shape[0]/(end_time_s - start_time_s))

    # Regarding the knn, method, dataset_name and distance choosen, set the file name to store the neighbors
    file_name = "./benchmarks/NearestNeighbors/" + dataset + "/knn_" + dataset + "_" + str(k) + "_" + distance + "_" + method + "_tg" + str(tam_grupo) + "_nc" + str(n_centroides) + "_r" + str(radio) + "_" + str(algorithm) + "_" + str(implementation) + ".hdf5"

    # Store indices, coords and dist into a hdf5 file
    save_neighbors(indices_vecinos, coords_vecinos, dists_vecinos, file_name)

    logging.info('------------------------------------------------------------------------\n')