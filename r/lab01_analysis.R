# ANLY 735 — Research Seminar in Predictive AI
# Replication Laboratory #1 | Evaluation Design Matters
# Student R Analysis
#
# Laboratory question:
# Does changing the evaluation design change estimated predictive
# performance—and does it change which model appears stronger?
#
# This script intentionally produces the analytical evidence.
# Interpretation, the FAIR audit, replication verdict, and claim boundaries
# belong in your replication-lab.qmd.
#
# Dataset:
# UCI Bike Sharing Dataset
# https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset


# ============================================================
# 1. REQUIRED PACKAGES
# ============================================================

required_packages <- c(
  "readr",
  "dplyr",
  "recipes",
  "parsnip",
  "workflows",
  "yardstick",
  "ranger",
  "glmnet",
  "iterators"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "\nRequired R packages are missing:\n  ",
      paste(missing_packages, collapse = ", "),
      "\n\nInstall them with:\n",
      "install.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "))\n"
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(recipes)
  library(parsnip)
  library(workflows)
  library(yardstick)
})


# ============================================================
# 2. REPRODUCIBILITY SETTINGS
# ============================================================

RANDOM_SEED <- 735
TEST_SIZE <- 0.20

set.seed(RANDOM_SEED)

# The script is designed to be run from the repository root:
#
# Rscript .\r\lab01_analysis.R
#
# or interactively from RStudio with the repository root as the
# working directory.

ROOT <- normalizePath(".", winslash = "/", mustWork = TRUE)

DATA_DIR <- file.path(ROOT, "data")
ANALYSIS_DIR <- file.path(ROOT, "analysis")

dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(ANALYSIS_DIR, showWarnings = FALSE, recursive = TRUE)

ZIP_PATH <- file.path(DATA_DIR, "Bike-Sharing-Dataset.zip")
CSV_PATH <- file.path(DATA_DIR, "hour.csv")

UCI_URL <- paste0(
  "https://archive.ics.uci.edu/ml/",
  "machine-learning-databases/00275/",
  "Bike-Sharing-Dataset.zip"
)


# ============================================================
# 3. ACQUIRE THE DATA REPRODUCIBLY
# ============================================================

if (!file.exists(CSV_PATH)) {

  cat("Downloading the UCI Bike Sharing Dataset...\n")

  tryCatch(
    {
      download.file(
        UCI_URL,
        destfile = ZIP_PATH,
        mode = "wb",
        quiet = FALSE
      )

      unzip(
        ZIP_PATH,
        files = "hour.csv",
        exdir = DATA_DIR
      )
    },
    error = function(e) {
      stop(
        paste0(
          "\nThe dataset could not be downloaded automatically.\n",
          "Check your internet connection and try again.\n",
          "Original error: ",
          conditionMessage(e)
        ),
        call. = FALSE
      )
    }
  )
}

cat("Using data:", normalizePath(CSV_PATH, winslash = "/"), "\n")


# ============================================================
# 4. LOAD DATA AND ESTABLISH CHRONOLOGY
# ============================================================

df <- read_csv(
  CSV_PATH,
  show_col_types = FALSE
)

df <- df |>
  mutate(
    datetime = as.POSIXct(
      paste(
        dteday,
        sprintf("%02d:00:00", hr)
      ),
      format = "%Y-%m-%d %H:%M:%S",
      tz = "UTC"
    )
  ) |>
  arrange(datetime)

cat("\nDataset dimensions:\n")
cat(nrow(df), "rows x", ncol(df), "columns\n")

cat("\nObservation period:\n")
cat(
  format(min(df$datetime), "%Y-%m-%d %H:%M:%S"),
  "to",
  format(max(df$datetime), "%Y-%m-%d %H:%M:%S"),
  "\n"
)


# ============================================================
# 5. DEFINE THE PREDICTION PROBLEM
# ============================================================

TARGET <- "cnt"

CATEGORICAL_FEATURES <- c(
  "season",
  "yr",
  "mnth",
  "hr",
  "holiday",
  "weekday",
  "workingday",
  "weathersit"
)

NUMERIC_FEATURES <- c(
  "temp",
  "atemp",
  "hum",
  "windspeed"
)

FEATURES <- c(
  CATEGORICAL_FEATURES,
  NUMERIC_FEATURES
)

# IMPORTANT:
# "casual" and "registered" are deliberately excluded.
# Their sum defines the target variable "cnt".
#
# "instant" is an identifier.
# "dteday" establishes chronology but is not used directly as a predictor.


# ============================================================
# 6. PREPARE MODELING DATA
# ============================================================

model_df <- df |>
  select(
    datetime,
    all_of(FEATURES),
    all_of(TARGET)
  )

# Convert categorical predictors to factors so that the R analysis
# treats them as categorical variables, matching the Python track's
# one-hot encoding strategy.

model_df <- model_df |>
  mutate(
    across(
      all_of(CATEGORICAL_FEATURES),
      as.factor
    )
  )


# ============================================================
# 7. DEFINE PREPROCESSING
# ============================================================

make_recipe <- function(training_data) {

  recipe(
    cnt ~ .,
    data = training_data |>
      select(-datetime)
  ) |>
    step_impute_median(all_numeric_predictors()) |>
    step_impute_mode(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
    step_normalize(all_numeric_predictors())
}


# ============================================================
# 8. DEFINE THE TWO COMPETING MODELS
# ============================================================

ridge_spec <- linear_reg(
  penalty = 1.0,
  mixture = 0
) |>
  set_engine("glmnet") |>
  set_mode("regression")

rf_spec <- rand_forest(
  trees = 300,
  min_n = 2
) |>
  set_engine(
    "ranger",
    seed = RANDOM_SEED,
    num.threads = max(1, parallel::detectCores() - 1)
  ) |>
  set_mode("regression")


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

evaluate_model <- function(
  model_spec,
  train_data,
  test_data,
  evaluation_design,
  model_name
) {

  rec <- make_recipe(train_data)

  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(model_spec)

  fitted_model <- fit(
    wf,
    data = train_data |> select(-datetime)
  )

  predictions <- predict(
    fitted_model,
    new_data = test_data |> select(-datetime)
  )

  evaluated <- bind_cols(
    test_data |> select(cnt),
    predictions
  )

  rmse_value <- rmse(
    evaluated,
    truth = cnt,
    estimate = .pred
  ) |>
    pull(.estimate)

  mae_value <- mae(
    evaluated,
    truth = cnt,
    estimate = .pred
  ) |>
    pull(.estimate)

  tibble(
    `Evaluation Design` = evaluation_design,
    Model = model_name,
    RMSE = rmse_value,
    MAE = mae_value
  )
}


# ============================================================
# 10. EXPERIMENT 1 — RANDOM EVALUATION
# ============================================================

set.seed(RANDOM_SEED)

n <- nrow(model_df)
test_n <- ceiling(n * TEST_SIZE)

test_indices <- sample(
  seq_len(n),
  size = test_n,
  replace = FALSE
)

train_random <- model_df[-test_indices, ]
test_random <- model_df[test_indices, ]

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("EXPERIMENT 1 — RANDOM EVALUATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")

cat("Training observations:", nrow(train_random), "\n")
cat("Testing observations :", nrow(test_random), "\n")

random_ridge <- evaluate_model(
  ridge_spec,
  train_random,
  test_random,
  "Random",
  "Ridge Regression"
)

random_rf <- evaluate_model(
  rf_spec,
  train_random,
  test_random,
  "Random",
  "Random Forest"
)


# ============================================================
# 11. EXPERIMENT 2 — TEMPORAL EVALUATION
# ============================================================

# Match the Python design:
# floor(80% of observations) are used for training and the
# remaining observations form the future test period.

split_index <- floor(n * (1 - TEST_SIZE))

train_temporal <- model_df[seq_len(split_index), ]
test_temporal <- model_df[(split_index + 1):n, ]

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("EXPERIMENT 2 — TEMPORAL EVALUATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")

cat("\nTraining period:\n")
cat(
  format(min(train_temporal$datetime), "%Y-%m-%d %H:%M:%S"),
  "to",
  format(max(train_temporal$datetime), "%Y-%m-%d %H:%M:%S"),
  "\n"
)

cat("\nTesting period:\n")
cat(
  format(min(test_temporal$datetime), "%Y-%m-%d %H:%M:%S"),
  "to",
  format(max(test_temporal$datetime), "%Y-%m-%d %H:%M:%S"),
  "\n"
)

cat("\nTraining observations:", nrow(train_temporal), "\n")
cat("Testing observations :", nrow(test_temporal), "\n")

temporal_ridge <- evaluate_model(
  ridge_spec,
  train_temporal,
  test_temporal,
  "Temporal",
  "Ridge Regression"
)

temporal_rf <- evaluate_model(
  rf_spec,
  train_temporal,
  test_temporal,
  "Temporal",
  "Random Forest"
)


# ============================================================
# 12. DISPLAY THE EVIDENCE
# ============================================================

results_df <- bind_rows(
  random_ridge,
  random_rf,
  temporal_ridge,
  temporal_rf
) |>
  mutate(
    RMSE = round(RMSE, 2),
    MAE = round(MAE, 2)
  )

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("ANLY 735 — EVALUATION RESULTS\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n", sep = "")

print(results_df)


# ============================================================
# 13. DISPLAY MODEL RANKINGS
# ============================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("MODEL RANKING BY RMSE\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")

for (design in c("Random", "Temporal")) {

  ranked <- results_df |>
    filter(`Evaluation Design` == design) |>
    arrange(RMSE)

  cat("\n", design, " evaluation:\n", sep = "")

  for (i in seq_len(nrow(ranked))) {
    cat(
      i,
      ". ",
      ranked$Model[i],
      " (RMSE = ",
      ranked$RMSE[i],
      ", MAE = ",
      ranked$MAE[i],
      ")\n",
      sep = ""
    )
  }
}


# ============================================================
# 14. SAVE REPRODUCIBLE RESULTS
# ============================================================

output_path <- file.path(
  ANALYSIS_DIR,
  "lab01_results.csv"
)

write_csv(
  results_df,
  output_path
)

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("RESULTS SAVED\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")

cat(normalizePath(output_path, winslash = "/"), "\n")


# ============================================================
# 15. RESEARCHER HANDOFF
# ============================================================

cat("\n", paste(rep("=", 60), collapse = ""), "\n", sep = "")
cat("YOUR JOB AS THE RESEARCHER\n")
cat(paste(rep("=", 60), collapse = ""), "\n", sep = "")

cat(
  "
The script has produced evidence. It has not interpreted that evidence for you.

Return to replication-lab.qmd and investigate:

1. How did RMSE and MAE change across evaluation designs?
2. Did the model ranking change?
3. Which conclusions remained stable?
4. Which conclusions require qualification?
5. Which evaluation design better represents the intended deployment scenario?
6. What does the evidence permit you to claim?
7. What does the evidence NOT permit you to claim?

Do not chase identical numbers. Investigate reproducibility.
"
)
