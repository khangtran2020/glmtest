#!/bin/bash

# This is a utility to retrieve all the potential modules from a list of project directories. 
#
# Each directory provided should be the root of the python modules.
#
# The list of project directories should contain one project directory per line. 

if [[ "$#" -lt 2 ]]; then
        echo "usage: $0 source_dir_lst out_file"
        exit 1
fi
src_dirs=$1
out=$2
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

while read -r source_dir; do python3 ${SCRIPT_DIR}/get_modules.py ${source_dir} | sed "s|^|${source_dir},|" ; done < $src_dirs > $out
