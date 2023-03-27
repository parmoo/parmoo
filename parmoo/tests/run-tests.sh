#!/bin/bash

# Get operation mode
export CHECK_PARMOO_SYNTAX=false
export UNIT_TESTS=false
export REGRESSION_TESTS=false
export LIST_COV_REPORT=false
while getopts :uclr flag; do
  case "${flag}" in
    (c) CHECK_PARMOO_SYNTAX=true;;
    (u) UNIT_TESTS=true;;
    (r) REGRESSION_TESTS=true;;
    (l) LIST_COV_REPORT=true;;
  esac
done
if [ $OPTIND -eq 1 ]; then
  export CHECK_PARMOO_SYNTAX=true;
  export UNIT_TESTS=true;
  export REGRESSION_TESTS=false;
  export LIST_COV_REPORT=true;
fi;

# Check style with flake8
if [ $CHECK_PARMOO_SYNTAX == true ]; then
  echo
  echo "Linting with flake8..."
  echo

  flake8 parmoo/*.py --per-file-ignores="__init__.py:F401";
  flake8 parmoo/tests/unit_tests/*.py;

  echo
  echo "Done."
  echo
fi;

# Run unit tests
if [ $UNIT_TESTS == true ]; then
  echo
  echo "Running unit tests with pytest and collecting coverage data..."
  echo
  python3 -m pytest -v --cov-config=parmoo/tests/.coveragerc --cov=parmoo --cov-report= parmoo/tests/unit_tests -W error::UserWarning;

  code=$? # capture pytest exit code
  if [ "$code" -eq "0" ]; then
    echo
    echo "Unit tests passed. Continuing..."
    echo
  else
    echo
    echo -e "Aborting run-tests.sh: Unit tests failed: $code"
    exit $code #return pytest exit code
  fi;

fi;

# Run regression tests
if [ $REGRESSION_TESTS == true ]; then

  # libE tests
  echo
  echo "Running libE tests (w/o pytest) and collecting coverage data..."
  echo
  python3 -m coverage run --rcfile=parmoo/tests/.coveragerc --parallel-mode --concurrency=multiprocessing parmoo/tests/libe_tests/test_libe_gen.py --comms local --nworkers 2

  code=$? # capture test_libe_gen exit code
  coverage combine --append

  if [ "$code" -eq "0" ]; then
    echo
    echo "libEnsemble tests passed. Continuing..."
    echo
  else
    echo
    echo -e "Aborting run-tests.sh: libEnsemble tests failed: $code"
    exit $code #return libE exit code
  fi;

  echo
  echo "Running regression tests with pytest and collecting coverage data..."
  echo
  for TEST_SCRIPT in parmoo/tests/regression_tests/test_*.py; do
    python3 -m coverage run --rcfile=parmoo/tests/.coveragerc --append $TEST_SCRIPT

    code=$? # capture pytest exit code
    if [ "$code" -eq "0" ]; then
      echo
      echo "$TEST_SCRIPT passed with code: $code"
      echo
    else
      echo
      echo "$TEST_SCRIPT failed with code: $code"
      echo -e "Aborting run-tests.sh: Regression tests failed: $code"
      exit $code #return pytest exit code
    fi;
  done;

  if [ "$code" -eq "0" ]; then
    echo
    echo "Regression tests passed. Continuing..."
    echo
  fi;
fi;

# Show coverage
if [ $LIST_COV_REPORT == true ]; then
  coverage report -m;
fi;
