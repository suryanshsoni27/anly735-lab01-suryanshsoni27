# Analysis Directory

This directory stores **generated analytical evidence** for Replication Laboratory #1.

After you successfully run either the Python or R analysis track, the script creates:

```text
lab01_results.csv
```

The file records:

- evaluation design;
- model;
- RMSE; and
- MAE.

## Source Code vs. Generated Evidence

The analysis scripts are the computational source:

```text
python/lab01_analysis.py
r/lab01_analysis.R
```

This directory contains evidence produced by running one of those scripts.

Students should **not manually edit `lab01_results.csv`** to obtain a preferred result.

If the analysis changes, rerun the script and allow the evidence file to be regenerated.

## Why Keep the Results File?

Unlike downloaded source data, `lab01_results.csv` should remain in your GitHub repository.

It provides a compact record of the evidence produced by your analytical environment and creates a traceable connection between:

```text
CODE → EXECUTION → EVIDENCE → INTERPRETATION → CLAIM
```

Your numerical results do not need to be identical to those produced by another programming language or software implementation.

The central reproducibility question is whether the analytical workflow is transparent enough to investigate why results agree or differ.

## Researcher Responsibility

The CSV contains results.

It does **not** contain the scientific conclusion.

Your interpretation belongs in:

```text
replication-lab.qmd
```

Use the results to complete the FAIR Model Comparison Audit, defend your replication verdict, and establish the boundaries of your research claim.

> **The script produces evidence. The researcher is responsible for the claim.**
