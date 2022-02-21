#!/bin/bash

# Set the libEnsemble path
#export LIBEPATH=/home/tyler/Git/libensemble
#export LIBEPATH=/Users/tyler/Git/libensemble

# Check style
flake8 unit_tests/*.py
flake8 ../parmoo/*.py

# Run unit tests
cd .. && export PYTHONPATH=$PYTHONPATH:`pwd` && cd tests
pytest -v --cov=../parmoo --cov-report= unit_tests -W error::UserWarning

# Run libE unit tests
#export PYTHONPATH=$PYTHONPATH:$LIBEPATH
#echo $PYTHONPATH
#python3 libe_tests/test_libe_gen.py --comms local --nworkers 4

coverage report -m
