"""Check generated evidence against the supplied report and record provenance.

Run python/lab01_analysis.py first. This verifier never changes the results CSV
or the report. It fails if the regenerated two-decimal metrics do not match.
"""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = (ROOT / "replication-lab.qmd").read_text(encoding="utf-8")
    expected = {}
    for line in report.splitlines():
        cells = [value.strip() for value in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] in {"Random", "Temporal"}:
            expected[(cells[0], cells[1])] = (float(cells[2]), float(cells[3]))
    if len(expected) != 4:
        raise ValueError("Expected four model/design result rows in the report.")

    results = pd.read_csv(ROOT / "analysis/lab01_results.csv")
    if len(results) != 4 or results.duplicated(["Evaluation Design", "Model"]).any():
        raise ValueError("The generated results must contain four distinct rows.")
    for _, row in results.iterrows():
        key = (row["Evaluation Design"], row["Model"])
        actual = (float(row["RMSE"]), float(row["MAE"]))
        if expected.get(key) != actual:
            raise ValueError(f"Report mismatch for {key}: {actual} vs {expected.get(key)}")

    for reference in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        if not (ROOT / reference).is_file():
            raise FileNotFoundError(f"Missing report figure: {reference}")
    with zipfile.ZipFile(ROOT / "replication-lab.docx") as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        docx_rows = []
        for row in document.findall(".//w:tr", ns):
            docx_rows.append([
                "".join(node.text or "" for node in cell.findall(".//w:t", ns)).strip()
                for cell in row.findall("w:tc", ns)
            ])
        for (design, model), (rmse, mae) in expected.items():
            if [design, model, f"{rmse:.2f}", f"{mae:.2f}"] not in docx_rows:
                raise ValueError(f"Word report table does not match: {design}, {model}")
        images = [name for name in archive.namelist() if name.startswith("word/media/")]
        figure = (ROOT / "analysis/model_performance.png").read_bytes()
        if not any(archive.read(name) == figure for name in images):
            raise ValueError("The repository figure differs from the report figure.")

    data = pd.read_csv(ROOT / "data/hour.csv")
    data["datetime"] = pd.to_datetime(data["dteday"] + " " + data["hr"].astype(str).str.zfill(2) + ":00:00")
    data = data.sort_values("datetime").reset_index(drop=True)
    random_train, random_test = train_test_split(data, test_size=0.20, random_state=735)
    cutoff = int(len(data) * 0.80)
    splits = {
        "random_train": random_train,
        "random_test": random_test,
        "temporal_train": data.iloc[:cutoff],
        "temporal_test": data.iloc[cutoff:],
    }
    split_summary = {
        name: {
            "observations": len(frame),
            "first_hour": str(frame["datetime"].min()),
            "last_hour": str(frame["datetime"].max()),
            "mean_hourly_rentals": round(float(frame["cnt"].mean()), 2),
        }
        for name, frame in splits.items()
    }
    expected_means = {
        "random_train": 189.57, "random_test": 189.02,
        "temporal_train": 174.64, "temporal_test": 248.75,
    }
    for name, mean in expected_means.items():
        if split_summary[name]["mean_hourly_rentals"] != mean:
            raise ValueError(f"Descriptive statistic in the report differs for {name}")

    files = [
        "python/lab01_analysis.py", "replication-lab.qmd", "replication-lab.docx",
        "analysis/lab01_results.csv", "analysis/model_performance.png", "data/hour.csv",
    ]
    record = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "template_commit": "fd8ce1bf8c4c02a8a5fbf2e65fb10802f9b6c86b",
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.machine(),
        "packages": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl"]},
        "report_metrics_match_generated_csv": True,
        "word_report_metrics_match_generated_csv": True,
        "figure_matches_original_word_report": True,
        "split_summary": split_summary,
        "results": results.to_dict(orient="records"),
        "sha256": {name: sha256(ROOT / name) for name in files},
    }
    output = ROOT / "analysis/reproduction_check.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("PASS: all eight metrics match the Quarto and Word reports.")
    print("PASS: report figure and descriptive split statistics match.")
    print("Recorded environment, data checksum, and evidence in analysis/reproduction_check.json")


if __name__ == "__main__":
    main()
