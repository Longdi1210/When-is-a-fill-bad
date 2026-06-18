.PHONY: reproduce test compile

reproduce:
	python3 scripts/run_main_analysis.py

test:
	PYTHONPATH=src python3 -m unittest discover tests

compile:
	python3 -m compileall src

.PHONY: real-btc real-btc-audit real-btc-convert real-btc-features real-btc-validation

real-btc: real-btc-audit real-btc-convert real-btc-features

real-btc-audit:
	PYTHONPATH=src python3 scripts/audit_kaggle_btc.py

real-btc-convert:
	PYTHONPATH=src python3 scripts/convert_kaggle_btc.py

real-btc-features:
	PYTHONPATH=src python3 scripts/build_real_btc_features.py

real-btc-validation:
	PYTHONPATH=src python3 scripts/run_real_btc_validation.py

.PHONY: dynamic-lob

dynamic-lob:
	PYTHONPATH=src python3 scripts/run_dynamic_lob_analysis.py
