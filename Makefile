INTERPRETER=python3
MAIN=src/main.py

run: $(MAIN)
	$(INTERPRETER) $(MAIN)
