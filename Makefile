.PHONY: check
check:
	ruff check --fix
	ruff format
	mypy agentlens tests example

.PHONY: test
test:
	pytest