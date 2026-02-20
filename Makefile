.PHONY: install install-dev run test cov

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

run:
	python app.py

test:
	pytest

cov:
	pytest --cov=app --cov-report=term-missing