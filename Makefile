PYTHON = python

-include admin_tasks
.PHONY: all_tests run_tests admin_tasks

# Run all tests in the tests directory
all_tests:
	$(PYTHON) -m unittest tests.gpflow_tests


# Define a variable for the test class
TEST_CLASS=

# Rule to execute the test
run_test:
	@if [ -z "$(TEST_CLASS)" ]; then \
		echo "Error: TEST_CLASS is not set. Use 'make run-test TEST_CLASS=<TestClassName>'"; \
		exit 1; \
	fi
	@echo "Running test class: $(TEST_CLASS)"
	@python -m unittest tests.gpflow_tests.$(TEST_CLASS)

# Default rule to display help
help:
	@echo "Usage:"
	@echo "  make run_test TEST_CLASS=<TestClassName>"
	@echo "    - Runs the specified test class using Python unittest"


# Perform all administrative tasks
admin_tasks:
	@echo "Running administrative tasks"
	@export PYTHONPATH=$(shell pwd):$$PYTHONPATH