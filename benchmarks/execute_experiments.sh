#!/bin/sh

# Script designed to launch a set of benchmarks based on .ini configuration files located in a specified directory.
#
# Usage:
#   ./execute_experiments.sh [/path/to/directory] [optional_filter]
#
# If no arguments are provided, the script will display a message indicating that a directory containing .ini files is required.
# If one argument is provided, the script will list all .ini files in the specified directory.
# If two arguments are provided, the script will filter the .ini files in the specified directory based on the search term.
#
# For each .ini file found or filtered, the script extracts the method name from the file name and executes a corresponding
# Python script to launch the experiment. If the method is "FLANN", it uses Python 2; otherwise, it uses Python 3.

case $# in

    1)
        config_files=$(ls "$1")
        ;;
    2)
        config_files=$(ls "$1" | grep "$2")
        ;;
    *)
        echo "Usage: ./execute_experiments.sh [/path/to/directory] [optional_filter]"
        exit 22
        ;;
esac

for file in $config_files
do
  echo $file

  method=$(echo "$file" | sed 's/.*_\(.*\)\.ini/\1/')

  if [ "$method" = "FLANN" ]
  then
    $(python2 -m experiments.test_launcher_py2 $file)
  else
    $(python3 -m experiments.test_launcher_py3 $file)
  fi
done

