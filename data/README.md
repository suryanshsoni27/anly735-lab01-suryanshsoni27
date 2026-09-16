# Data Directory

This directory is used for the source data required by **Replication Laboratory #1 — Evaluation Design Matters**.

## Dataset

The laboratory uses the **UCI Bike Sharing Dataset**, containing hourly bike-rental observations from 2011–2012.

Dataset documentation:

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

The hourly data file used by the laboratory is:

```text
hour.csv
```

## Reproducible Data Acquisition

You do **not** need to manually download the dataset.

Both analysis tracks are designed to obtain the original dataset automatically when `data/hour.csv` is not already available.

### Python

```text
python ./python/lab01_analysis.py
```

### R

Run the R script from the repository root, for example:

```r
source("r/lab01_analysis.R")
```

The scripts download and extract the required UCI data into this directory.

## Why the Raw Data Are Not Committed

The downloaded dataset and ZIP archive are excluded from Git through `.gitignore`.

This is intentional.

A reproducible repository does not necessarily need to store a copy of every external dataset. When a stable public source is available, preserving the **data source, acquisition procedure, and analytical code** can provide a clearer provenance trail while avoiding unnecessary duplication.

For this laboratory, another researcher should be able to clone the repository, run the selected analysis script, and reconstruct the required local data.

## Information Legitimacy

The outcome is:

```text
cnt
```

The analysis intentionally excludes `casual` and `registered` as predictors because those variables are components of `cnt`.

The variable `instant` is an identifier, and `dteday` is used to establish chronological order rather than as a direct predictor.

> **Reproducibility includes documenting not only what data were used, but why particular information was or was not permitted to enter the model.**
