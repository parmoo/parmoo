#!/bin/bash

# Get operation mode
export CHECK_PARMOO_SYNTAX=false
export RUN_PARMOO_TEST=false
export SHOW_PARMOO_COV=false
while getopts :tcs flag; do
  case "${flag}" in
    (c) CHECK_PARMOO_SYNTAX=true;;
    (t) RUN_PARMOO_TEST=true;;
    (l) SHOW_PARMOO_COV=true;;
  esac
done
if [ $OPTIND -eq 1 ]; then
  export CHECK_PARMOO_SYNTAX=true;
  export RUN_PARMOO_TEST=true;
  export SHOW_PARMOO_COV=true;
fi;

# Check style with flake8
if [ $CHECK_PARMOO_SYNTAX == true ]; then
  flake8 parmoo/*.py --per-file-ignores="__init__.py:F401";
  flake8 parmoo/tests/unit_tests/*.py;
fi;

# Run unit tests
if [ $RUN_PARMOO_TEST == true ]; then
  pytest -v --cov-config=parmoo/tests/.coverageac --cov=parmoo --cov-report= parmoo/tests/unit_tests -W error::UserWarning;

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

# Show coverage
if [ $SHOW_PARMOO_COV == true ]; then
  coverage report -m;
fi;
