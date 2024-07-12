from benchmarks.neighbors_utils import *
from mask.experiments.test_knn.main_test_knn_radius_config import *

# Set gaussian clouds parameters
nclouds = 8
overlap = [False, True]
npc = [200]
same_train_test = False

# Set parameters for benchmarks
k = 5  # Busqueda puntual (k=1 -> 1 punto = primer vecino)
distance = 'euclidean'
method = 'MASK'
tg = 16
nc = 8
r = 35

#rep = 1

### REPLICATE EXPERIMENTS FROM PAPER 1 (EXHAUSTIVE POINT QUERY - Table 3 & 4) USING FLANN #####

# Set log configuration
logging.basicConfig(filename='./2023_InfoSci/logs/exhaustive_pointer_search_MASK.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
logging.info('------------------------------------------------------------------------')
logging.info('                   EXHAUSTIVE POINTER SEARCH USING MASK')
logging.info('------------------------------------------------------------------------\n')

for o in overlap:
    for n in npc:
        logging.info(' ')
        logging.info('--------- Gaussian Clouds with %s points per cloud and %s overlap-------', str(n), str(o))

        gaussian_clouds = "gaussian_clouds_npc" + str(n) + "_" + str(o)
        mask_conf = str(k) + "_" + distance + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r)

        #print("test_knn_" + synthetic_data + "_" + mask_conf + ".ini")
        MASK("test_knn_" + gaussian_clouds + "_" + mask_conf + ".ini")

        logging.info('-------------------------- Error rate  ------------------------------')
        # Exhaustive Point Query Error Rate
        file_name_le = "./benchmarks/NearestNeighbors/gaussian-clouds/knn_" + gaussian_clouds + "_" + str(k) + "_" + distance + "_BruteForce.hdf5"
        file_name_mc = "./benchmarks/NearestNeighbors/gaussian-clouds/knn_" + gaussian_clouds + "_" + mask_conf + ".hdf5"
        print("\n")
        er = error_rate(gaussian_clouds, distance, method, k, same_train_test, file_name_le, file_name_mc)
