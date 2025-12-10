# Makefile for fraud anomaly detection project
# Helps run the pipeline and regenerate figures

ENV_NAME = fraud-anomaly
PYTHON = python

FIG_DIR = fig
RESULTS_DIR = results

# -----------------------
# Environment
# -----------------------

env:
	conda env create -f environment.yml

activate:
	@echo "Run this to activate the environment:"
	@echo "conda activate $(ENV_NAME)"

# -----------------------
# Run project
# -----------------------

run:
	$(PYTHON) run_pipeline.py --mode run

tune:
	$(PYTHON) run_pipeline.py --mode tune

plots:
	$(PYTHON) analysis.py 

# -----------------------
# Cleanup
# -----------------------

clean:
	rm -rf $(FIG_DIR) $(RESULTS_DIR)

# -----------------------
# Help
# -----------------------

	@echo ""
	@echo "Available commands:"
	@echo "  make env    Create conda environment"
	@echo "  make run    Run final pipeline"
	@echo "  make tune   Run hyperparameter tuning"
	@echo "  make clean  Remove results"
	@echo ""