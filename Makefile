INTERPRETER=python3
MAIN=src/main.py
TESTDIR=src/test

run: $(MAIN)
	$(INTERPRETER) $(MAIN)

test:
	$(INTERPRETER) -m unittest discover -s $(TESTDIR)

