from data.load_train_test_set import *

# Set gaussian clouds parameters
nclouds = 8
overlap = [False, True]
npc = [200]

# Set parameters for benchmarks
k = 5  # Busqueda puntual (k=1 -> 1 punto = primer vecino)
distance = 'euclidean'
method = 'BruteForce'

rep = 1
algorithm = 'brute'


### REPLICATE EXPERIMENTS FROM PAPER 1 (EXHAUSTIVE POINT QUERY - Table 3 & 4) USING BruteForce #####

# Set log configuration
logging.basicConfig(filename='./2023_InfoSci/logs/exhaustive_pointer_search_BruteForce.log',
                    filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
logging.info('------------------------------------------------------------------------')
logging.info('               EXHAUSTIVE POINTER SEARCH USING BRUTE FORCE')
logging.info('------------------------------------------------------------------------\n')

# Build gaussian datasets with different configurations and store them
logging.info('------------------- Gaussian Clouds Generation ---------------------\n')

for o in overlap:
    for n in npc:

        gaussian_clouds = "gaussian_clouds_npc" + str(n) + "_" + str(o)
        load_train_test(gaussian_clouds, test_eq_train=True)


logging.info(' ')
logging.info('-------------------------- Pointer Search  ------------------------------')
for o in overlap:
    for n in npc:
        logging.info(' ')
        logging.info('--------- Gaussian Clouds with %s points per cloud and %s overlap-------', str(n), str(o))

        gaussian_clouds = "gaussian_clouds_npc" + str(n) + "_" + str(o)
        bruteforce_conf = str(k) + "_" + distance + "_" + method

        Exact("test_knn_" + gaussian_clouds + "_" + bruteforce_conf + ".ini")

        # Dataset indexation and exhaustive Point Query using BruteForce (knn=1)
        # index_time, search_time = knn_bruteforce_modified(synthetic_data)

        # logging.info('Index time= %s seconds - %s rep iterations average', index_time, rep)
        # logging.info('Search time= %s seconds - %s rep iterations average\n\n', search_time, rep)

