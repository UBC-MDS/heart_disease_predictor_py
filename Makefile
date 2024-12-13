.PHONY: clean
# Directories
RESULTS_DIR = results
FIGURES_DIR = $(RESULTS_DIR)/figures
MODELS_DIR = $(RESULTS_DIR)/models
TABLES_DIR = $(RESULTS_DIR)/tables

# Directories
REPORT_DIR = report
REPORT_FILES_DIR = $(REPORT_DIR)/heart_disease_predictor_report_files

# File paths
QMD_FILE = $(REPORT_DIR)/heart_disease_predictor_report.qmd
HTML_FILE = $(REPORT_DIR)/heart_disease_predictor_report.html
PDF_FILE = $(REPORT_DIR)/heart_disease_predictor_report.pdf
BIB_FILE = $(REPORT_DIR)/references.bib

# File paths
EDA_FIGURES = $(FIGURES_DIR)/eda_output_categorical_features_distribution.png \
              $(FIGURES_DIR)/eda_output_categorical_stacked_barplots.png \
              $(FIGURES_DIR)/eda_output_numeric_boxplots.png \
              $(FIGURES_DIR)/eda_output_raw_feature_distributions.png \
              $(FIGURES_DIR)/eda_output_summary_stats.csv \
              $(FIGURES_DIR)/eda_output_target_distribution.png \
              $(FIGURES_DIR)/log_reg_feature_coefficients.png \
              $(FIGURES_DIR)/raw_boxplots_by_class.png \
              $(FIGURES_DIR)/raw_correlation_heatmap.png \
              $(FIGURES_DIR)/raw_feature_distributions.png \
              $(FIGURES_DIR)/target_variable_distribution.png

MODELS = $(MODELS_DIR)/heart_disease_Lr_pipeline.pickle \
         $(MODELS_DIR)/heart_disease_preprocessor.pickle \
         $(MODELS_DIR)/heart_disease_svc_pipeline.pickle

TABLES = $(TABLES_DIR)/baseline_cv_results.csv \
         $(TABLES_DIR)/best_model_cv_results.csv \
         $(TABLES_DIR)/coefficient_df.csv \
         $(TABLES_DIR)/misclassified_examples.csv \
         $(TABLES_DIR)/test_score.csv

# Default target
all: $(HTML_FILE) $(PDF_FILE) $(REPORT_FILES_DIR)

# Target for generating figures (assuming scripts or commands to create them)
# $(FIGURES_DIR)/eda_output_categorical_features_distribution.png:
# 	python scripts/SCRIPT_NAME \
# 		--input_file=INPUT_PATH \
# 		--output_file=$(FIGURES_DIR)/eda_output_categorical_features_distribution.png

$(MODELS_DIR)/heart_disease_preprocessor.pickle $(TABLES_DIR)/heart_disease_test.csv \
$(TABLES_DIR)/heart_disease_train.csv $(TABLES_DIR)/scaled_heart_disease_test.csv \
$(TABLES_DIR)/scaled_heart_disease_train.csv: scripts/split_n_preprocess.py data/raw/processed.cleveland.data
	python scripts/split_n_preprocess.py \
		--input-path=data/raw/processed.cleveland.data \
		--data-dir=data/processed \
		--preprocessor-dir=$(MODELS_DIR) \
		--seed=522

$(HTML_FILE) $(PDF_FILE) $(REPORT_FILES_DIR) : $(RESULTS_DIR) $(BIB_FILE)
	quarto render $(QMD_FILE)

# $(PDF_FILE): $(QMD_FILE) $(BIB_FILE)
#     quark $(QMD_FILE) --to pdf --output $(PDF_FILE) --bibliography $(BIB_FILE)

# Clean generated files
clean:
	rm -f $(EDA_FIGURES) $(MODELS) $(TABLES)
	rm -rf $(REPORT_FILES_DIR)
	rm -rf data/processed
