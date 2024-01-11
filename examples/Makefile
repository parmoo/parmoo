OUT_FILES = quickstart.out named_var_ex.out unnamed_var_ex.out \
			precomputed_data.out checkpointing.out advanced_ex.out \
			local_method.jpeg libe_basic_ex.out quickstart.jpeg

all: $(OUT_FILES)

quickstart.jpeg: quickstart.py
	python3 quickstart.py > quickstart.out
	mv Pareto\ Front.jpeg quickstart.jpeg

quickstart.out: quickstart.py
	python3 quickstart.py > quickstart.out
	mv Pareto\ Front.jpeg quickstart.jpeg

named_var_ex.out: named_var_ex.py
	python3 named_var_ex.py > named_var_ex.out

unnamed_var_ex.out: unnamed_var_ex.py
	python3 unnamed_var_ex.py > unnamed_var_ex.out

precomputed_data.out: precomputed_data.py
	python3 precomputed_data.py > precomputed_data.out

checkpointing.out: checkpointing.py
	python3 checkpointing.py > checkpointing.out
	rm -f parmoo.moop parmoo.surrogate.1 parmoo.search.1 parmoo.optimizer

advanced_ex.out: advanced_ex.py
	python3 advanced_ex.py > advanced_ex.out
	rm -f parmoo.moop parmoo.surrogate.1 parmoo.acquisition.2 parmoo.search.1
	rm -f parmoo.optimizer

local_method.jpeg: local_method.py
	python3 local_method.py
	mv Pareto\ Front.jpeg local_method.jpeg

libe_basic_ex.out: libe_basic_ex.py
	python3 libe_basic_ex.py --comms local --nworkers 3 > libe_basic_ex.out
	rm -f parmoo.moop parmoo.surrogate.1 parmoo.search.1 parmoo.optimizer
	rm -f libE_stats.txt ensemble.log

clean:
	rm -f *.out libE_stats.txt ensemble.log *.jpeg
