# ===========================
# TomeWeaver Makefile
# ===========================

# Virtual environment path
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# ===========================
# Setup
# ===========================

# Create virtual environment
venv:
	python3 -m venv $(VENV)

# Install dependencies
install: venv
	$(PIP) install -r requirements.txt

# ===========================
# Run TomeWeaver
# ===========================

# Run the full TomeWeaver pipeline
run:
	$(PYTHON) -m tomeweaver

# Optional: Regenerate TOC only (if you add generate_toc.py)
toc:
	$(PYTHON) generate_toc.py

# ===========================
# Utilities
# ===========================

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Remove virtualenv
clean_venv:
	rm -rf $(VENV)

# Reinstall everything cleanly
reinstall: clean clean_venv install
