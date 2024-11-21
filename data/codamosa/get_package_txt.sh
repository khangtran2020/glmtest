#!/bin/bash

# This is a utility to retrieve the package dependencies from a list of project directories,
# and store the directories in a file named package.txt in the project directory.
# 
# Each directory provided should be the root of the python modules.
#
# The list of project directories should contain one project directory per line.


if [[ "$#" -lt 1 ]]; then
        echo "usage: $0 source_dir_lst"
        exit 1
fi
src_dirs=$1

while read -r source_dir; do
        echo "pipreqs --savepath ${source_dir}/package.txt $source_dir"
        pipreqs --savepath ${source_dir}/package.txt --ignore ${source_dir}/test,${source_dir}/tests $source_dir
done < $src_dirs