INTERPRETER=python3
MAIN=src/main.py
TOPLEVELDIR=src

.PHONY: run
run: $(MAIN)
	$(INTERPRETER) $(MAIN)

.PHONY: test
test:
	$(INTERPRETER) -m unittest discover $(TOPLEVELDIR)

