INTERPRETER=python3
MAIN=src/main.py
TOPLEVELDIR=src

run: $(MAIN)
	$(INTERPRETER) $(MAIN)

test:
	$(INTERPRETER) -m unittest discover $(TOPLEVELDIR)

