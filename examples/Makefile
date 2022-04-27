OUT_FILES = quickstart.out named_var_ex.out unnamed_var_ex.out \
			precomputed_data.out advanced_ex.out libe_basic_ex.out

all: $(OUT_FILES)

quickstart.out: quickstart.py
	python3 quickstart.py > quickstart.out

named_var_ex.out: named_var_ex.py
	python3 named_var_ex.py > named_var_ex.out

unnamed_var_ex.out: unnamed_var_ex.py
	python3 unnamed_var_ex.py > unnamed_var_ex.out

precomputed_data.out: precomputed_data.py
	python3 precomputed_data.py > precomputed_data.out

advanced_ex.out: advanced_ex.py
	python3 advanced_ex.py > advanced_ex.out
	rm -f parmoo.moop parmoo.surrogate.1

libe_basic_ex.out: libe_basic_ex.py
	python3 libe_basic_ex.py --comms local --nworkers 3 > libe_basic_ex.out
	rm -f parmoo.moop parmoo.surrogate.1 libE_stats.txt ensemble.log

clean:
	rm -f *.out libE_stats.txt ensemble.log
