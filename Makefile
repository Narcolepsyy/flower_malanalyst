DATASET := Obfuscated-MalMem2022.csv
DATASET_SHA256 := 624ecb6c9229cf62bd3b8f65d23a2b132760e6bada458aa46eee134a313840c4

.PHONY: data train experiments dashboard test clean

data:
	kaggle datasets download -d hasanccr92/cic-malmem-2022 -p /tmp/flmal-data --unzip
	cp /tmp/flmal-data/$(DATASET) $(DATASET)
	printf '%s  %s\n' '$(DATASET_SHA256)' '$(DATASET)' | sha256sum -c -

train:
	python run_single_experiment.py --preset quick --model logreg --agg-method fedavg

experiments:
	python run_experiments.py --preset dev

dashboard:
	python dashboard_interactive.py --host 0.0.0.0 --port 8503

test:
	python -m unittest discover -s tests

clean:
	rm -rf ray_results .pytest_cache .mypy_cache .ruff_cache __pycache__ federated_malware/__pycache__ tests/__pycache__
