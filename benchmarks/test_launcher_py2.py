from benchmarks.algorithms.FLANN.knn import FLANN

import re
import argparse
import ConfigParser
import io

def main(args):

    # Get the path of the configuration file provided by the user
    config_file = args.config_file
    dataset = re.split('_|\.', config_file)[2]
    method = re.split('_|\.',  config_file)[5]
    configfile_path = "./benchmarks/config/" + dataset + "/" + config_file
    print("--- Reading " + config_file + " ---")

    # Open the configuration file
    with open(configfile_path) as f:
        config_file = f.read()
    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.readfp(io.BytesIO(config_file))

    # According to the method choosen, carry out the experiment described on the configuration file

    if method == 'BruteForce':
        print("Please, use test_launcher_py3")

    elif method == 'Exact':
        print("Please, use test_launcher_py3")

    elif method == 'GDASC':
        print("Please, use test_launcher_py3")

    elif method == 'FLANN':
        FLANN(config)

    elif method == 'PYNN':
        print("Please, use test_launcher_py3")

    else:
        print("Method not able")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Config file | .ini", type=str)
    args = parser.parse_args()

    main(args)
