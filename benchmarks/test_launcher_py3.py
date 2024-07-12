from benchmarks.algorithms.Exact.knn import Exact
from benchmarks.algorithms.Pynndescent.knn import PYNN
from benchmarks.algorithms.GDASC.knn import GDASC
import re
import argparse

def main(args):

    config_file = args.config_file
    method = re.split('_|\.', config_file)[5]

    # According to the method chosen, carry out the experiment described on the configuration file
    if method == 'Exact':
        Exact(config_file)

    elif method == 'GDASC':
        GDASC(config_file)

    elif method == 'FLANN':
        print("Please, use test_launcher_py2")

    elif method == 'PYNN':
        PYNN(config_file)

    else:
        print("Method not able")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Config file | .ini", type=str)
    args = parser.parse_args()

    main(args)
