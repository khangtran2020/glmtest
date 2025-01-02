#!/usr/bin/env bash

INPUT_DIR="/test"
PROJECT_DIR="/project"
OUTPUT_DIR="/output"
PACKAGE_DIR="/package"

function error_echo {
  RED="\033[0;31m"
  NC="\033[0m"
  echo -e "${RED}ERROR: ${1}${NC}\n"
}

if [[ ! -d ${INPUT_DIR} || -z "$(ls -A ${INPUT_DIR})" ]]
then
  error_echo "You need to specify a mount to ${INPUT_DIR}"
  exit 1
fi

# Check if the /output mount point is present
if [[ ! -d ${OUTPUT_DIR} ]]
then
  error_echo "You need to specify a mount to ${OUTPUT_DIR}"
  exit 1
fi

# Check if the /package mount point is present
if [[ ! -d ${PACKAGE_DIR} && ! -f ${PACKAGE_DIR}/package.txt ]]
then
  error_echo "You need to specify a mount to ${PACKAGE_DIR} containing package.txt"
  exit 1
fi

pip install -r "${PACKAGE_DIR}/package.txt"
cd /project

echo "Command is: cp -r ${INPUT_DIR}/$1 ."
cp -r "${INPUT_DIR}"/"$1" .

echo "Variables is: $1 $2"

coverage run --branch --data-file "$2" -m pytest "$1"
coverage report

mv "$2" /output
rm test_case_*.py