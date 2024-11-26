PYTHON = python

.PHONY: gpflow_tests

gpflow_tests:
	@export PYTHONPATH=$(shell pwd):$$PYTHONPATH && \
	$(PYTHON) -m unittest discover -s tests -p "gpflow_tests.py"

# Run all tests in the tests directory
all_tests:
	@export PYTHONPATH=$(shell pwd):$$PYTHONPATH && \
	$(PYTHON) -m unittest discover -s tests
