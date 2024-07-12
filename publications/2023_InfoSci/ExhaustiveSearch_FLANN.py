from benchmarks.neighbors_utils import *

# Set gaussian clouds parameters
overlap = [False, True]
npc = [200, 1000]

# Set parameters for benchmarks
method = 'FLANN'
distance = 'euclidean'
k = 1  # Busqueda puntual (1 punto = primer vecino)

### REPLICATE EXPERIMENTS FROM PAPER 1 (EXHAUSTIVE POINT QUERY - Table 3 & 4) USING FLANN #####

# Set log configuration
logging.basicConfig(filename='.2023_InfoSci/logs/exhaustive_pointer_search_FLANN.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
logging.info('------------------------------------------------------------------------')
logging.info('                   EXHAUSTIVE POINTER SEARCH USING FLANN')
logging.info('------------------------------------------------------------------------\n')
logging.info(' ')
for o in overlap:
    for n in npc:
        logging.info(' ')
        logging.info('--------- Gaussian Clouds with %s points per cloud and %s overlap-------', str(n), str(o))

        gaussian_clouds = "gaussian_clouds_npc" + str(n) + "_" + str(o)
        flann_conf = str(k) + "_" + distance + "_" + method

        # Dataset indexation and exhaustive Point Query using FLANN (knn=1)
        FLANN("test_knn_" + gaussian_clouds + "_" + flann_conf + ".ini")

        # Exhaustive Point Query Error Rate
        file_name_le = "./2023_InfoSci/neighbors/BruteForce/knn_" + gaussian_clouds + ".hdf5"
        file_name_mc = "./2023_InfoSci/neighbors/FLANN/knn_" + gaussian_clouds + ".hdf5"

        er = error_rate(gaussian_clouds, distance, 'FLANN', k, file_name_le, file_name_mc)