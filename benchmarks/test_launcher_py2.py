from benchmarks.algorithms.FLANN.knn import FLANN

import re
import argparse


def main(args):

    config_file = args.config_file
    method = re.split('_|\.', config_file)[5]

    # According to the method chosen, carry out the experiment described on the configuration file
    if method == 'Exact':
        print("Please, use test_launcher_py3")

    elif method == 'GDASC':
        print("Please, use test_launcher_py3")

    elif method == 'FLANN':
        FLANN(config_file)

    elif method == 'PYNN':
        print("Please, use test_launcher_py3")

    else:
        print("Method not able")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Config file | .ini", type=str)
    args = parser.parse_args()

    main(args)
