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
	conda create --name py311 -c conda-forge python=3.11 && conda activate py311

sec-install:
	conda install -c conda-forge --file requirements.txt

test:
	python -m pytest -vv test_MyRuleRankingLib.py -W ignore::DeprecationWarning

docker-lint:
	hadolint Dockerfile

lint:
	pylint --disable=R,C test_MyRuleRankingLib_beta.py

all: install lint test