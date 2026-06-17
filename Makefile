.PHONY: reproduce test compile

reproduce:
	python3 scripts/run_main_analysis.py

test:
	PYTHONPATH=src python3 -m unittest discover tests

compile:
	python3 -m compileall src

