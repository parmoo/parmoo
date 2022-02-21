# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = python3 -msphinx
SOURCEDIR     = .
BUILDDIR      = ./build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile $(EX_OUT)
	cd ../examples && python3 quickstart.py > quickstart.out
	cd ../examples && python3 named_var_ex.py > named_var_ex.out
	cd ../examples && python3 unnamed_var_ex.py > unnamed_var_ex.out
	cd ../examples && python3 precomputed_data.py > precomputed_data.out
	cd ../examples && python3 advanced_ex.py > advanced_ex.out
	cd ../examples && python3 libe_basic_ex.py --comms local --nworkers 3 \
		> libe_basic_ex.out
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
