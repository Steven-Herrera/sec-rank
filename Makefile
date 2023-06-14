install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

env:
	python3 -m venv ~/.sec-rank && source ~/.sec-rank/bin/activate

format:
	black *.py

sec-env-init:
	conda init bash

sec-env:
	conda create --name py311 -c conda-forge python=3.11 && conda init bash
	#conda activate py311

sec-install:
	conda config --set ssl_verify no && conda install -c conda-forge --file requirements.txt
	pip install -r sec_pip_requirements.txt
	pip install --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

test:
	python3 -m pytest -vv tests/*.py -W ignore::DeprecationWarning

test-dag:
	python3 -m pytest -vv test_airbyte_airflow_sec_dag.py

test-webscraper:
	python3 -m pytest -vv test_webscraper.py

docker-lint:
	hadolint Dockerfile

lint:
	pylint --disable=R,C MyRuleRankingLib.py

lint-webscraper:
	pylint --disable=R,C test_webscraper.py

all: install lint test