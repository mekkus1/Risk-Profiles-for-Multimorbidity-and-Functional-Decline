############################################################
# MIDUS Refresher 2: Machine Learning Analysis
# Mapping Shared Risk Profiles of Multimorbidity and
# Functional Decline in U.S. Adults
#
# Author: Philips N. Okeagu
# Contact: philipsokeagu@hsph.harvard.edu
#
# Repository: https://github.com/mekkus1/Risk-Profiles-for-Multimorbidity-and-Functional-Decline
# Date: 2026
#
# Description:
# This script performs supervised machine learning analysis
# to identify risk clusters for multimorbidity and functional
# decline using MIDUS Refresher 2 data.
#
# Requirements: See README.md for package versions
############################################################

# =========================
# 0. Environment Setup
# =========================

# Clear workspace
rm(list = ls())

# Set seed for reproducibility
set.seed(123)

# Create output directories if they don't exist
if(!dir.exists("output")) dir.create("output")
if(!dir.exists("figures")) dir.create("figures")
if(!dir.exists("tables")) dir.create("tables")

# =========================
# 1. Load Required Packages
# =========================

required_packages <- c(
  "tidyverse", "haven", "labelled", "missForest", "caret",
  "glmnet", "randomForest", "xgboost", "pROC", "SHAPforxgboost",
  "janitor", "ggplot2", "gridExtra", "knitr"
)

# Install missing packages
new_packages <- required_packages[
  !(required_packages %in% installed.packages()[,"Package"])
]
if(length(new_packages)) install.packages(new_packages)

# Load packages
invisible(lapply(required_packages, library, character.only = TRUE))

# Package versions for reproducibility
cat("\n=== R Package Versions ===\n")
cat("R version:", R.version.string, "\n")
cat("tidyverse:", as.character(packageVersion("tidyverse")), "\n")
cat("caret:", as.character(packageVersion("caret")), "\n")
cat("randomForest:", as.character(packageVersion("randomForest")), "\n")
cat("xgboost:", as.character(packageVersion("xgboost")), "\n")
cat("glmnet:", as.character(packageVersion("glmnet")), "\n")
cat("pROC:", as.character(packageVersion("pROC")), "\n")

# =========================
# 2. Data Loading
# =========================

#' Load MIDUS Refresher 2 SPSS Data
#'
#' @param file_path Path to the SPSS .sav file
#' @return Cleaned dataframe with numeric variables

load_midus_data <- function(file_path) {

  cat("\n=== Loading MIDUS Refresher 2 Data ===\n")

  if(!file.exists(file_path)) {
    stop("Data file not found at: ", file_path)
  }

  midus_raw <- read_sav(file_path)

  midus <- midus_raw %>%
    clean_names() %>%
    mutate(across(
      where(~inherits(., "haven_labelled")),
      ~as.numeric(as.character(.))
    ))

  cat("Data loaded successfully\n")
  cat("Sample size:", nrow(midus), "participants\n")
  cat("Number of variables:", ncol(midus), "\n")

  return(midus)
}

# Update this path to your data location
# For users: update the path below (e.g., "data/MR2_P1_SURVEY_N2154_20251003 (1).sav")
data_path <- ""

# Check if here package is available for path management
if(requireNamespace("here", quietly = TRUE)) {
  library(here)
  cat("Using 'here' package for path management\n")
  data_path <- here("data", "MR2_P1_SURVEY_N2154_20251003 (1).sav")
}

midus <- load_midus_data(data_path)

# =========================
# 3. Apply Variable Labels
# =========================

midus <- midus %>%
  set_variable_labels(
    # Demographics
    rb1prage = "Age (years)",
    rb1prsex = "Biological sex",
    rb1pf7a = "Race/Ethnicity",
    rb1pb1 = "Education level",
    rb1pb16 = "Income",
    rb1pb19 = "Marital status",
    rb1sf17b = "Employment status",

    # Chronic Conditions
    rb1sa11s = "Hypertension",
    rb1sa11x = "Diabetes",
    rb1sa12d = "Heart condition medication",
    rb1pa6a = "Stroke",
    rb1pa26 = "Cancer",
    rb1sa11c = "Lung disease",
    rb1sa11d = "Joint/Bone disease",
    rb1pa2 = "Mental/Emotional disorder",

    # Functional Limitations
    rb1sa24b = "Bathing/Dressing limitation",
    rb1sa24f = "Walking >1 mile limitation",
    rb1sa24c = "Climb stairs limitation",
    rb1sa24af = "Housework limitation",
    rb1sa24ag = "Managing money limitation",
    rb1pd12 = "Shopping/cooking/housework limitation",
    rb1sbadl1 = "Basic ADL limitation",
    rb1pa1 = "Self-rated physical health",

    # Mental Health
    rb1pa60 = "Depressive episode ≥2 weeks",
    rb1pg100b = "Loneliness",
    rb1sa20b = "Nervous frequency",
    rb1sa20d = "Hopeless frequency",
    rb1sa20f = "Worthless frequency",

    # Sleep
    rb1sa53a = "Sleep hours",
    rb1sa57a = "Trouble falling asleep",
    rb1sa57b = "Night waking",
    rb1sa57d = "Unrested during day",

    # Body
    rb1sa33a = "Height (feet)",
    rb1sa35 = "Weight (pounds)",
    rb1sbmi = "BMI",
    rb1sa31 = "Waist circumference",

    # Stress
    rb1se1z = "Feeling overwhelmed",
    rb1se4e = "Life beyond control",
    rb1sp1e = "Fired times",

    # Health Behaviors
    rb1pa39 = "Current smoker",
    rb1pa40 = "Cigarettes per day",
    rb1pa51 = "Alcohol use frequency",
    rb1pa55 = "Drinks per occasion",
    rb1sa52f = "Exercise frequency",

    # Social Support
    rb1se1p = "Few close friends",
    rb1sh10a = "Emotional support spouse",
    rb1sh10b = "Emotional support parents",
    rb1sh10c = "Emotional support in-laws",
    rb1sh10d = "Emotional support children",
    rb1sh10e = "Emotional support others",
    rb1sl1 = "Living with partner",
    rb1slfedi = "Lifetime discrimination"
  )

# =========================
# 4. Create Outcome Variables
# =========================

cat("\n=== Creating Outcome Variables ===\n")

# Chronic conditions for multimorbidity
disease_vars <- c(
  "rb1sa11s", "rb1sa11x", "rb1sa12d", "rb1pa6a",
  "rb1pa26", "rb1sa11c", "rb1sa11d", "rb1pa2"
)

# Validated ADL/IADL measures for functional decline
true_func_vars <- c("rb1sbadl1", "rb1pd12")

midus <- midus %>%
  mutate(
    # Multimorbidity (≥2 chronic conditions)
    disease_count = rowSums(midus[, disease_vars] == 1, na.rm = TRUE),
    multimorbidity = ifelse(disease_count >= 2, 1, 0),

    # Functional decline - at least ONE limitation in ADL/IADL (score ≥2)
    func_decline_true = ifelse(
      rowSums(midus[, true_func_vars] >= 2, na.rm = TRUE) >= 1,
      1, 0
    ),

    # Secondary outcomes for sensitivity analyses
    any_adl = ifelse(rb1sbadl1 > 1, 1, 0),
    any_iadl = ifelse(rb1pd12 > 1, 1, 0)
  )

# Display outcome prevalence
cat("\n=== Outcome Prevalence ===\n")
cat("Multimorbidity:",
    round(mean(midus$multimorbidity, na.rm = TRUE) * 100, 1), "%\n")
cat("Functional Decline (ADL/IADL):",
    round(mean(midus$func_decline_true, na.rm = TRUE) * 100, 1), "%\n")
cat("  - Any ADL limitation:",
    round(mean(midus$any_adl, na.rm = TRUE) * 100, 1), "%\n")
cat("  - Any IADL limitation:",
    round(mean(midus$any_iadl, na.rm = TRUE) * 100, 1), "%\n")

# =========================
# 5. Define Predictor Set
# =========================

predictors <- c(
  # Mental Health
  "rb1pa60", "rb1sa20b", "rb1sa20d", "rb1sa20f", "rb1pg100b",
  # Sleep
  "rb1sa53a", "rb1sa57a", "rb1sa57b", "rb1sa57d",
  # Body
  "rb1sbmi", "rb1sa31",
  # Stress
  "rb1se1z", "rb1se4e", "rb1sp1e",
  # SES
  "rb1sg1", "rb1pb1", "rb1pb16", "rb1sf17b", "rb1sc1",
  # Behavior - REMOVED rb1pa40 (93% missing)
  "rb1pa39", "rb1pa51", "rb1pa55", "rb1sa52f",
  # Social
  "rb1se1p", "rb1pb19", "rb1slfedi",
  # Demographics
  "rb1prage", "rb1prsex", "rb1pf7a"
)

# Create analysis dataset
vars_to_keep <- c("multimorbidity", "func_decline_true", predictors)
vars_to_keep <- vars_to_keep[vars_to_keep %in% names(midus)]
ml_data <- midus[, vars_to_keep]

# Rename for modeling consistency
names(ml_data)[names(ml_data) == "func_decline_true"] <- "functional_decline"

# =========================
# 6. Handle High Missingness Variables
# =========================

cat("\n=== Handling High Missingness Variables ===\n")

# Recode smoking status to include "Unknown" category
# Original coding: 1=Yes, 2=No
ml_data <- ml_data %>%
  mutate(
    rb1pa39 = case_when(
      is.na(rb1pa39) ~ 3,
      rb1pa39 == 1 ~ 1,
      rb1pa39 == 2 ~ 2,
      TRUE ~ 3
    )
  )

cat("Smoking status recoded: 1=Yes, 2=No, 3=Unknown\n")
cat("Distribution:\n")
print(table(ml_data$rb1pa39, useNA = "ifany"))

# Convert character to factor
char_cols <- sapply(ml_data, is.character)
ml_data[char_cols] <- lapply(ml_data[char_cols], as.factor)

cat("\n=== Analysis Dataset Created ===\n")
cat("Dimensions:", nrow(ml_data), "rows,", ncol(ml_data), "columns\n")

# =========================
# 7. Missing Data Imputation
# =========================

cat("\n=== Missing Data Summary ===\n")
missing_pct <- sapply(ml_data, function(x) mean(is.na(x)) * 100)
print(round(missing_pct[missing_pct > 0], 1))

#' Simple median/mode imputation
#' @param x Vector to impute
#' @return Imputed vector
impute_simple <- function(x) {
  if(is.numeric(x)) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
  } else if(is.factor(x)) {
    mode_val <- names(sort(table(x), decreasing = TRUE))[1]
    x[is.na(x)] <- mode_val
  }
  return(x)
}

ml_data <- as.data.frame(lapply(ml_data, impute_simple))

# Convert factors to numeric for modeling
factor_cols <- sapply(ml_data, is.factor)
if(any(factor_cols)) {
  for(col in names(ml_data)[factor_cols]) {
    ml_data[[col]] <- as.numeric(ml_data[[col]])
  }
}

cat("\nMissing data handled\n")

# =========================
# 8. Train/Test Split
# =========================

set.seed(42)
train_index <- createDataPartition(
  ml_data$multimorbidity,
  p = 0.7,
  list = FALSE
)
train <- ml_data[train_index, ]
test <- ml_data[-train_index, ]

cat("\n=== Sample Sizes ===\n")
cat("Training set: n =", nrow(train), "\n")
cat("Test set: n =", nrow(test), "\n")

# =========================
# 9. Model Evaluation Function
# =========================

#' Evaluate model performance
#' @param actual True binary outcomes
#' @param predicted Predicted probabilities
#' @param model_name Name of model
#' @param outcome Name of outcome
#' @return Data frame with performance metrics

evaluate_model <- function(actual, predicted, model_name, outcome) {

  roc_obj <- roc(actual, predicted, quiet = TRUE)
  auc_val <- auc(roc_obj)

  youden <- coords(roc_obj, "best",
                   ret = c("threshold", "sensitivity", "specificity"))

  pred_class <- ifelse(predicted > youden$threshold[1], 1, 0)

  cm <- confusionMatrix(
    as.factor(pred_class),
    as.factor(actual),
    positive = "1"
  )

  brier <- mean((predicted - actual)^2)

  data.frame(
    Outcome = outcome,
    Model = model_name,
    AUC = round(auc_val, 3),
    Sensitivity = round(youden$sensitivity[1], 3),
    Specificity = round(youden$specificity[1], 3),
    Threshold = round(youden$threshold[1], 3),
    Accuracy = round(cm$overall["Accuracy"], 3),
    Brier = round(brier, 3),
    stringsAsFactors = FALSE
  )
}

# =========================
# 10. Multimorbidity Models
# =========================

cat("\n=== Training Multimorbidity Models ===\n")

# 10.1 Logistic Regression
base_model_mm <- glm(multimorbidity ~ ., data = train, family = "binomial")
base_pred_mm <- predict(base_model_mm, newdata = test, type = "response")

# 10.2 LASSO Regression
x_train_mm <- as.matrix(train[, !names(train) %in% "multimorbidity"])
y_train_mm <- train$multimorbidity
x_test_mm <- as.matrix(test[, !names(test) %in% "multimorbidity"])

set.seed(123)
cv_lasso_mm <- cv.glmnet(
  x_train_mm, y_train_mm,
  family = "binomial", alpha = 1, nfolds = 5
)
lasso_model_mm <- glmnet(
  x_train_mm, y_train_mm,
  family = "binomial", alpha = 1, lambda = cv_lasso_mm$lambda.min
)
lasso_pred_mm <- predict(lasso_model_mm, x_test_mm, type = "response")[,1]

# 10.3 Random Forest
set.seed(123)
rf_model_mm <- randomForest(
  as.factor(multimorbidity) ~ .,
  data = train, ntree = 500, importance = TRUE
)
rf_pred_mm <- predict(rf_model_mm, test, type = "prob")[,2]

# 10.4 XGBoost
feature_names_mm <- colnames(train[, !names(train) %in% "multimorbidity"])

xgb_train_mm <- xgb.DMatrix(
  data = as.matrix(train[, feature_names_mm]),
  label = train$multimorbidity
)
xgb_test_mm <- xgb.DMatrix(
  data = as.matrix(test[, feature_names_mm]),
  label = test$multimorbidity
)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model_mm <- xgb.train(
  params = params,
  data = xgb_train_mm,
  nrounds = 100,
  verbose = 0
)
xgb_pred_mm <- predict(xgb_model_mm, xgb_test_mm)

# Evaluate multimorbidity models
results_mm <- bind_rows(
  evaluate_model(test$multimorbidity, base_pred_mm, "Logistic", "Multimorbidity"),
  evaluate_model(test$multimorbidity, lasso_pred_mm, "LASSO", "Multimorbidity"),
  evaluate_model(test$multimorbidity, rf_pred_mm, "Random Forest", "Multimorbidity"),
  evaluate_model(test$multimorbidity, xgb_pred_mm, "XGBoost", "Multimorbidity")
)

print(results_mm)

# =========================
# 11. Functional Decline Models
# =========================

cat("\n=== Training Functional Decline Models ===\n")

# 11.1 Logistic Regression
base_model_fd <- glm(functional_decline ~ ., data = train, family = "binomial")
base_pred_fd <- predict(base_model_fd, newdata = test, type = "response")

# 11.2 LASSO Regression
x_train_fd <- as.matrix(train[, !names(train) %in% "functional_decline"])
y_train_fd <- train$functional_decline
x_test_fd <- as.matrix(test[, !names(test) %in% "functional_decline"])

set.seed(123)
cv_lasso_fd <- cv.glmnet(
  x_train_fd, y_train_fd,
  family = "binomial", alpha = 1, nfolds = 5
)
lasso_model_fd <- glmnet(
  x_train_fd, y_train_fd,
  family = "binomial", alpha = 1, lambda = cv_lasso_fd$lambda.min
)
lasso_pred_fd <- predict(lasso_model_fd, x_test_fd, type = "response")[,1]

# 11.3 Random Forest
set.seed(123)
rf_model_fd <- randomForest(
  as.factor(functional_decline) ~ .,
  data = train, ntree = 500, importance = TRUE
)
rf_pred_fd <- predict(rf_model_fd, test, type = "prob")[,2]

# 11.4 XGBoost
feature_names_fd <- colnames(train[, !names(train) %in% "functional_decline"])

xgb_train_fd <- xgb.DMatrix(
  data = as.matrix(train[, feature_names_fd]),
  label = train$functional_decline
)
xgb_test_fd <- xgb.DMatrix(
  data = as.matrix(test[, feature_names_fd]),
  label = test$functional_decline
)

xgb_model_fd <- xgb.train(
  params = params,
  data = xgb_train_fd,
  nrounds = 100,
  verbose = 0
)
xgb_pred_fd <- predict(xgb_model_fd, xgb_test_fd)

# Evaluate functional decline models
results_fd <- bind_rows(
  evaluate_model(test$functional_decline, base_pred_fd, "Logistic", "Functional Decline"),
  evaluate_model(test$functional_decline, lasso_pred_fd, "LASSO", "Functional Decline"),
  evaluate_model(test$functional_decline, rf_pred_fd, "Random Forest", "Functional Decline"),
  evaluate_model(test$functional_decline, xgb_pred_fd, "XGBoost", "Functional Decline")
)

print(results_fd)

# =========================
# 12. Combined Results
# =========================

all_results <- bind_rows(results_mm, results_fd)

cat("\n=== Final Model Performance ===\n")
print(all_results %>% arrange(Outcome, desc(AUC)))

# Save results
write.csv(all_results, "tables/table2_model_performance.csv", row.names = FALSE)

# =========================
# 13. Feature Importance
# =========================

cat("\n\n========================================\n")
cat("FEATURE IMPORTANCE\n")
cat("========================================\n")

# Random Forest Importance
rf_imp_mm <- as.data.frame(importance(rf_model_mm))
rf_imp_mm$Variable <- rownames(rf_imp_mm)
rf_imp_mm <- rf_imp_mm[order(-rf_imp_mm$MeanDecreaseGini), ]

rf_imp_fd <- as.data.frame(importance(rf_model_fd))
rf_imp_fd$Variable <- rownames(rf_imp_fd)
rf_imp_fd <- rf_imp_fd[order(-rf_imp_fd$MeanDecreaseGini), ]

# XGBoost Importance
xgb_imp_mm <- xgb.importance(model = xgb_model_mm, feature_names = feature_names_mm)
xgb_imp_fd <- xgb.importance(model = xgb_model_fd, feature_names = feature_names_fd)

# LASSO Coefficients
lasso_coef_mm <- as.matrix(coef(lasso_model_mm))
lasso_coef_mm <- data.frame(
  Variable = rownames(lasso_coef_mm),
  Coefficient = lasso_coef_mm[,1]
)
lasso_coef_mm <- lasso_coef_mm[
  lasso_coef_mm$Coefficient != 0 &
    lasso_coef_mm$Variable != "(Intercept)",
]
lasso_coef_mm <- lasso_coef_mm[order(-abs(lasso_coef_mm$Coefficient)), ]

lasso_coef_fd <- as.matrix(coef(lasso_model_fd))
lasso_coef_fd <- data.frame(
  Variable = rownames(lasso_coef_fd),
  Coefficient = lasso_coef_fd[,1]
)
lasso_coef_fd <- lasso_coef_fd[
  lasso_coef_fd$Coefficient != 0 &
    lasso_coef_fd$Variable != "(Intercept)",
]
lasso_coef_fd <- lasso_coef_fd[order(-abs(lasso_coef_fd$Coefficient)), ]

# Save importance files
write.csv(rf_imp_mm, "tables/rf_importance_multimorbidity.csv", row.names = FALSE)
write.csv(rf_imp_fd, "tables/rf_importance_functional_decline.csv", row.names = FALSE)
write.csv(xgb_imp_mm, "tables/xgb_importance_multimorbidity.csv", row.names = FALSE)
write.csv(xgb_imp_fd, "tables/xgb_importance_functional_decline.csv", row.names = FALSE)
write.csv(lasso_coef_mm, "tables/lasso_coefficients_multimorbidity.csv", row.names = FALSE)
write.csv(lasso_coef_fd, "tables/lasso_coefficients_functional_decline.csv", row.names = FALSE)

# =========================
# 14. SHAP Values
# =========================

cat("\n\n=== Calculating SHAP Values ===\n")

shap_mm <- shap.values(
  xgb_model = xgb_model_mm,
  X_train = as.matrix(train[, feature_names_mm])
)

shap_fd <- shap.values(
  xgb_model = xgb_model_fd,
  X_train = as.matrix(train[, feature_names_fd])
)

mean_shap_mm <- data.frame(
  Variable = colnames(shap_mm$shap_score),
  Mean_SHAP = colMeans(abs(shap_mm$shap_score))
) %>% arrange(desc(Mean_SHAP))

mean_shap_fd <- data.frame(
  Variable = colnames(shap_fd$shap_score),
  Mean_SHAP = colMeans(abs(shap_fd$shap_score))
) %>% arrange(desc(Mean_SHAP))

write.csv(mean_shap_mm, "tables/shap_importance_multimorbidity.csv", row.names = FALSE)
write.csv(mean_shap_fd, "tables/shap_importance_functional_decline.csv", row.names = FALSE)

# =========================
# 15. Generate Session Info
# =========================

sink("output/session_info.txt")
cat("Analysis Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
cat("=== SESSION INFO ===\n\n")
print(sessionInfo())
sink()

cat("\n=== Analysis Complete ===\n")
cat("Results saved to 'tables/' directory\n")
cat("Session info saved to 'output/session_info.txt'\n")
