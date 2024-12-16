.PHONY: clean

# Directories
RESULTS_DIR = results
FIGURES_DIR = $(RESULTS_DIR)/figures
MODELS_DIR = $(RESULTS_DIR)/models
TABLES_DIR = $(RESULTS_DIR)/tables
REPORT_DIR = report
REPORT_FILES_DIR = $(REPORT_DIR)/heart_disease_predictor_report_files

# File paths
QMD_FILE = $(REPORT_DIR)/heart_disease_predictor_report.qmd
HTML_FILE = $(REPORT_DIR)/heart_disease_predictor_report.html
PDF_FILE = $(REPORT_DIR)/heart_disease_predictor_report.pdf
BIB_FILE = $(REPORT_DIR)/references.bib

# Figures, models, and tables
EDA_FIGURES = $(FIGURES_DIR)/eda_output_categorical_features_distribution.png \
              $(FIGURES_DIR)/eda_output_categorical_stacked_barplots.png \
              $(FIGURES_DIR)/eda_output_numeric_boxplots.png \
              $(FIGURES_DIR)/eda_output_raw_feature_distributions.png \
              $(FIGURES_DIR)/eda_output_target_distribution.png
ANALYSIS_FIGURES = $(FIGURES_DIR)/log_reg_feature_coefficients.png
MODELS = $(MODELS_DIR)/heart_disease_lr_pipeline.pickle \
         $(MODELS_DIR)/heart_disease_preprocessor.pickle \
         $(MODELS_DIR)/heart_disease_svc_pipeline.pickle
TABLES = $(TABLES_DIR)/eda_output_summary_stats.csv \
         $(TABLES_DIR)/baseline_cv_results.csv \
         $(TABLES_DIR)/best_model_cv_results.csv \
         $(TABLES_DIR)/coefficient_df.csv \
         $(TABLES_DIR)/misclassified_examples.csv \
         $(TABLES_DIR)/test_score.csv

# Default target
all: $(HTML_FILE) $(PDF_FILE) $(REPORT_FILES_DIR)

# Generate results directory and subdirectories
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR) $(FIGURES_DIR) $(MODELS_DIR) $(TABLES_DIR)

# Download raw data
data/raw/heart_disease.zip data/raw/processed.cleveland.data: | data/raw
	python scripts/download_data.py \
		--url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip" \
		--path="data/raw"

data/raw:
	mkdir -p data/raw

# Preprocessing
data/processed/heart_disease_train.csv: scripts/split_n_preprocess.py data/raw/processed.cleveland.data
	python scripts/split_n_preprocess.py \
		--input-path=data/raw/processed.cleveland.data \
		--data-dir=data/processed \
		--preprocessor-dir=$(MODELS_DIR) \
		--seed=522

# Generate EDA figures and tables
$(EDA_FIGURES) $(TABLES_DIR)/eda_output_summary_stats.csv: scripts/script_eda.py data/processed/heart_disease_train.csv
	python scripts/script_eda.py \
		--input_data_path=data/processed/heart_disease_train.csv \
		--output_prefix=$(RESULTS_DIR)

# Train models
$(TABLES_DIR)/baseline_cv_results.csv $(TABLES_DIR)/best_model_cv_results.csv $(MODELS): data/processed/heart_disease_train.csv $(MODELS_DIR)/heart_disease_preprocessor.pickle
	python scripts/fit_heart_disease_predictor.py \
		--train-set=data/processed/heart_disease_train.csv \
		--preprocessor=$(MODELS_DIR)/heart_disease_preprocessor.pickle \
		--pipeline-to=$(MODELS_DIR) \
		--table-to=$(TABLES_DIR) \
		--seed=522

# Evaluate models
$(TABLES_DIR)/test_score.csv $(TABLES_DIR)/coefficient_df.csv $(TABLES_DIR)/misclassified_examples.csv $(ANALYSIS_FIGURES): data/processed/heart_disease_test.csv $(MODELS)
	python scripts/evaluate_heart_disease_predictor.py \
		--test-set=data/processed/heart_disease_test.csv \
		--pipeline-svc-from=$(MODELS_DIR)/heart_disease_svc_pipeline.pickle \
		--pipeline-lr-from=$(MODELS_DIR)/heart_disease_lr_pipeline.pickle \
		--table-to=$(TABLES_DIR) \
		--plot-to=$(FIGURES_DIR) \
		--seed=522

# Render report
$(HTML_FILE) $(PDF_FILE) $(REPORT_FILES_DIR): $(TABLES) $(MODELS) $(EDA_FIGURES) $(ANALYSIS_FIGURES) $(BIB_FILE)
	quarto render $(QMD_FILE)

# Clean generated files
clean:
	rm -f $(EDA_FIGURES) $(ANALYSIS_FIGURES) $(MODELS) $(TABLES) $(HTML_FILE) $(PDF_FILE)
	rm -rf $(REPORT_FILES_DIR) data/processed data/raw
