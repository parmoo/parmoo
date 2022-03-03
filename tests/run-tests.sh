#!/bin/bash

# Set the libEnsemble path
#export LIBEPATH=/home/tyler/Git/libensemble
#export LIBEPATH=/Users/tyler/Git/libensemble

# Check style with flake8
flake8 unit_tests/*.py --per-file-ignores="__init__.py:F401"
flake8 ../parmoo/*.py --per-file-ignores="__init__.py:F401"

# Run unit tests
pytest -v --cov=../parmoo --cov-report= unit_tests # -W error::UserWarning

code=$? # capture pytest exit code
if [ "$code" -eq "0" ]; then
  echo
  tput bold;tput setaf 2; echo "Unit tests passed. Continuing...";tput sgr 0
  echo
else
  echo
  tput bold;tput setaf 1;echo -e "Aborting run-tests.sh: Unit tests failed: $code";tput sgr 0
  exit $code #return pytest exit code
fi;

# Run libE unit tests
#export PYTHONPATH=$PYTHONPATH:$LIBEPATH
#echo $PYTHONPATH
#python3 libe_tests/test_libe_gen.py --comms local --nworkers 4

coverage report -m
