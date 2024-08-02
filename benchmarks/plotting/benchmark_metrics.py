from benchmarks.neighbors_utils import *
import logging
import pandas as pd



# Recall Benchmark
def recall(dataset_name, d, method, k, same_train_test=False, file_name_le=None, file_name=None):

    # Recall in Exhaustive Point Query (query points are the same from training set)
    if same_train_test:

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            if indices_mc[i] == i:
                hit = hit + 1
        '''

        # Count number of 1-neighbor which are the same as the point searched
        #hit = map(lambda x, y: x == y, list(indices_mc), range(indices_mc.shape[0])).count(True)


    # Recall in query points different from training set
    else:
        # Load neighbors obtained through linear exploration
        indices_le, coords_le, dists_le = load_neighbors(file_name_le)

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            hit = hit + len(np.intersect1d(indices_mc[i].astype(int), indices_le[i]))
        '''

        # Count number of 1-neighbor which are the same as the point searched
        hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices_mc), list(indices_le)))

    # Recall: %  hit returned vs number of points
    rec = hit / indices_mc.size * 100


    # Show percentage of hit/miss on screen and save information on log file
    '''
    print ("---- Case " + str(k) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Hit percentage: " + str(rec) + "%\n\n")
    logging.info("---- Case " + str(k) + " nn applying " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    '''
    logging.info("Correct neighbors rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    logging.info("Hit percentage: " + str(rec) + "%\n\n")

    return rec


# Error rate
def error_rate(dataset_name, d, method, knn, same_train_test=False, file_name_le=None, file_name=None):

    # Error rate in Exhaustive Point Query when query points are the same from training set
    if same_train_test:

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            if indices_mc[i] == i:
                hit = hit + 1
        '''

        # Count number of 1-neighbor which are the same as the point searched
        hit = map(lambda x, y: x == y, list(indices_mc), range(indices_mc.shape[0])).count(True)

    # Error rate in Exhaustive Point Query when query points are the same from training set
    else:
        # Load neighbors obtained through linear exploration
        indices_le, coords_le, dists_le = load_neighbors(file_name_le)

        # Load neighbors obtained through the method choosen
        indices_mc, coords_mc, dists_mc = load_neighbors(file_name)

        '''
        hit = 0.0
        for i in range(indices_mc.shape[0]):
            hit = hit + len(np.intersect1d(indices_mc[i].astype(int), indices_le[i]))
        '''

        # Count number of 1-neighbor which are the same as the point searched
        # We set assume_unique=True at np.intersectid(...) to accelerate the calculation
        # bc the compared lists are always uniques (any element would ever appear twice as neighbor for the same point)
        hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y, assume_unique=True)), list(indices_mc), list(indices_le)))

    # Compare: % miss returned vs number of points
    er = (1 - hit / float(indices_mc.size)) * 100

    # Show percentage of hit/miss on screen an save information on a log file
    '''print("---- Case " + str(knn) + " nn within " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Found points rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    print("Error rate: " + str(er) + "%\n\n")
    logging.info("")
    logging.info("---- Case " + str(knn) + " nn within " + method + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    '''
    logging.info("Found points rate: " + str(hit) + "/" + str(float(indices_mc.size)))
    logging.info("Error rate: " + str(er) + "%")

    return er


# Compare intersection percentage between neighbors found by two different methods
def compare(dataset_name, d, method1, method2, knn, file_name1=None, file_name2=None):

    # Load neighbors obtained through first method
    indices_m1, coords_m1, dists_m1 = load_neighbors(file_name1)

    # Load neighbors obtained through the second method choosen
    indices_m2, coords_m2, dists_m2 = load_neighbors(file_name2)

    # Count number of 1-neighbor which are calculated as the same by both methods
    hit = sum(map(lambda x, y: len(np.intersect1d(x.astype(int), y)), list(indices_m2), list(indices_m1)))


    # Compare: %  hit returned vs number of points
    ip = hit/indices_m2.size * 100

    # Show percentage of hit/miss on screen an save information on a log file
    print ("---- Case " + str(knn) + " nn within " + method1 + " and " + method2 + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    print("Same neighbors rate: " + str(hit) + "/" + str(float(indices_m2.size)))
    print("Intersection percentage: " + str(ip) + "%\n\n")
    logging.info("---- Case " + str(knn) + " nn within " + method1 + " and " + method2 + " over " + str(dataset_name) + " dataset using " + str(d) + " distance. ----")
    logging.info("Same neighbors rate: " + str(hit) + "/" + str(float(indices_m2.size)))
    logging.info("Intersection percentage: " + str(ip) + "%\n\n")

    return ip


def get_error_rate(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline):

    for da in datasets:

        # GDASC configurations
        if da == 'NYtimes':
            tg = 1000
            nc = 500
            r = 30

        elif da == 'GLOVE':
            tg = 1000
            nc = 500
            r = 250

        elif da == 'MNIST':
            tg = 1000
            nc = 500
            r = 80000

        elif da == 'municipios':
            tg = 60
            nc = 30
            r = 40

        elif da == 'wdbc':
            tg = 50
            nc = 25
            r = 7500

        # Set logging info
        logging.basicConfig(
            filename='./benchmarks/logs/' + da + '/' + da + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(gdasc_algorithm) + "" + str(gdasc_implementation) + '_errorrate.log',
            filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO)
        logging.info('------------------------------------------------------------------------')
        logging.info('                       plotting over %s Dataset ERROR RATE', da)
        logging.info('------------------------------------------------------------------------')
        logging.info('Search of k neighbors over a choosen dataset, using different methods, run_benchmarks.py')
        logging.info('------------------------------------------------------------------------\n')

        logging.info('Distances: %s ', distances)
        logging.info('Methods: %s', methods)
        logging.info('plotting: %s', knn)
        logging.info('GDASC params: tg=%s, nc=%s, r=%s, algorithm=%s, implementation=%s\n', tg, nc, r, gdasc_algorithm, gdasc_implementation)

        # From a chosen dataset, calculate recalls, store them and print graph
        da_error_rate = []
        for di in distances:
            logging.info('------------  %s distance  --------------------', di)
            di_error_rate = []
            for method in methods:
                m_error_rate = []

                logging.info('')
                logging.info('-- %s method --', method)
                logging.info('')

                for k in knn:

                    file_name_le = "./benchmarks/NearestNeighbors/" + da + "/knn_" + da + "_" + str(k) + "_" + di + "_" + baseline + ".hdf5"

                    if method == 'GDASC':
                        file_name = "./benchmarks/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(
                            r) + "_" + str(gdasc_algorithm) + "_" + str(gdasc_implementation) + ".hdf5"

                    else:
                        file_name = "./benchmarks/NearestNeighbors/" + da + "/knn_" + da + "_" + str(k) + "_" + di + "_" + method + ".hdf5"

                    er = error_rate(da, di, method, k, False, file_name_le, file_name)

                    m_error_rate.append(er)

                di_error_rate.append(m_error_rate)
            da_error_rate.append(di_error_rate)

    return da_error_rate

def get_recall(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline):

    recalls = pd.DataFrame(columns=['Dataset', 'k', 'Distance', 'Method', 'Recall'])

    for da in datasets:

        # GDASC configurations
        if da == 'NYtimes':
            tg = 1000
            nc = 500
            r = 30

        elif da == 'GLOVE':
            tg = 1000
            nc = 500
            r = 250

        elif da == 'MNIST':
            tg = 1000
            nc = 500
            r = 80000

        elif da == 'municipios':
            tg = 60
            nc = 30
            r = 40

        elif da == 'wdbc':
            tg = 50
            nc = 25
            r = 7500

        # Set logging info
        logging.basicConfig(filename='./benchmarks/logs/' + da + '/' + da + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + "GDASC" + "." + str(gdasc_implementation) +'_recall.log',
                            filemode='w', format='%(asctime)s - %(name)s - %(message)s', level=logging.INFO, force=True)
        logging.info('------------------------------------------------------------------------')
        logging.info('                            %s Dataset RECALL', da)
        logging.info('------------------------------------------------------------------------')
        logging.info('Search of k nearest neighbors over a choosen dataset, using different methods, run_benchmarks.py')
        logging.info('------------------------------------------------------------------------\n')

        logging.info('Distances: %s ', distances)
        logging.info('Methods: %s', methods)
        logging.info('plotting: %s', knn)
        logging.info('GDASC params: tg=%s, nc=%s, r=%s, algorithm=%s, implementation=%s\n', tg, nc, r, gdasc_algorithm, gdasc_implementation)

        # From a chosen dataset, calculate recall for each benchmarks  (k-dataset-distance-method combination)
        for di in distances:
            logging.info('------------  %s distance  --------------------\n', di)
            for method in methods:

                logging.info('-- %s method --\n', method)

                for k in knn:

                    file_name_le = "./benchmarks/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                        k) + "_" + di + "_" + baseline + ".hdf5"

                    if method == 'GDASC':
                        file_name = "./benchmarks/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + "_tg" + str(tg) + "_nc" + str(nc) + "_r" + str(r) + "_" + str(gdasc_algorithm) + "_" + str(gdasc_implementation) + ".hdf5"
                    else:
                        file_name = "./benchmarks/NearestNeighbors/" + da + "/knn_" + da + "_" + str(
                            k) + "_" + di + "_" + method + ".hdf5"

                    if not os.path.isfile(file_name):
                        re = np.nan
                    else:
                        re = recall(da, di, method, k, False, file_name_le, file_name)

                    # And store each into a pandas DataFrame
                    recalls = pd.concat([recalls, pd.DataFrame([{'Dataset': da, 'k': k, 'Distance': di, 'Method': method, 'Recall': re}])], ignore_index=True)
                    #print(recalls)

    logging.shutdown()
    return recalls

def get_avgRecall(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline):

    # Once we have obtained the recall for each experiment (each k-dataset-distance-method combination)
    recalls = get_recall(datasets, distances, methods, knn, gdasc_algorithm, gdasc_implementation, baseline)

    # Obtain mean Average Points for each dataset-distance-method combination
    avgRecall = recalls.groupby(['Dataset', 'Distance', 'Method'])['Recall'].mean().reset_index()
    #print("average Recall (avgRecall):\n\n " + str(avgRecall))
    print(avgRecall)

    return avgRecall
