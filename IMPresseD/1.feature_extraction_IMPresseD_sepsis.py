# 0) Setup

# Imports
import os
import sys
import csv
import glob
import shutil
import importlib
from functools import reduce
from typing import List, Dict

import pandas as pd
import yaml
import pm4py
import subprocess

import helper_functions.IMPresseD.IMPresseD_No_GUI as imp_nom

# Add repository root to PYTHONPATH for local imports
PACKAGE_ROOT = os.path.abspath(os.getcwd())
sys.path.insert(0, PACKAGE_ROOT)

# Load configuration
with open(os.path.join(PACKAGE_ROOT, "config", "config_feature_extraction_IMPresseD_sepsis.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Paths and constants
PROJECT_ROOT = PACKAGE_ROOT
EVENT_LOG    = cfg["event_log"]
RAW_LOG      = os.path.join(PROJECT_ROOT, "0_raw_log", cfg["raw_log"])
UNIQUE_DIR   = os.path.join(PROJECT_ROOT, "1_unique_log")
LABELED_DIR  = os.path.join(PROJECT_ROOT, "2_labelled_logs", EVENT_LOG)
FEATURES_DIR = os.path.join(PROJECT_ROOT, "3_extracted_features", EVENT_LOG)

SPLIT_RATIO   = 0.7
SEQ_THRESHOLD = 5
MAX_SPLITS    = 1
settings      = cfg["event_log_settings"]
exp_name      = cfg["experiment_name"]

OUTPUT_PATH = os.path.join(FEATURES_DIR, f"{exp_name}_features", "IMPresseD")

# Ensure output directories exist
os.makedirs(UNIQUE_DIR,  exist_ok=True)
os.makedirs(LABELED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"Event log: {EVENT_LOG}")
print(f"Settings: {settings}")
print(f"Experiment: {exp_name}")

# Configure IMPresseD CLI arguments (in-process entry)
sys.argv = [
    "IMPresseD_No_GUI.py",
    "--log_path",            f"2_labelled_logs/{EVENT_LOG}/{exp_name}.csv",
    "--output_path",         f"3_extracted_features/{EVENT_LOG}/{exp_name}_features/IMPresseD",
    "--discovery_type",      "auto",
    "--case_id",             "case:concept:name",
    "--activity",            "concept:name",
    "--timestamp",           "time:timestamp",
    "--outcome",             "case:Label",
    "--outcome_type",        "binary",
    "--delta_time",          "1",
    "--max_gap",             "5",
    "--max_extension_step",  "3",
    "--testing_percentage",  "0.001",
    "--extension_style",     "All",
    "--numerical_attributes", "Age", "Leucocytes", "CRP", "LacticAcid",
    "--categorical_attributes",
        "InfectionSuspected", "org:group", "DiagnosticBlood", "DisfuncOrg",
        "SIRSCritTachypnea", "Hypotensie", "SIRSCritHeartRate", "Infusion",
        "DiagnosticArtAstrup", "DiagnosticIC", "DiagnosticSputum",
        "DiagnosticLiquor", "DiagnosticOther", "SIRSCriteria2OrMore",
        "DiagnosticXthorax", "SIRSCritTemperature", "DiagnosticUrinaryCulture",
        "SIRSCritLeucos", "Oligurie", "DiagnosticLacticAcid", "Diagnose",
        "Hypoxie", "DiagnosticUrinarySediment", "DiagnosticECG",
]

# 1) Convert labeled XES logs to CSV
for xes_path in glob.glob(os.path.join(LABELED_DIR, "*.xes")):
    log = pm4py.read_xes(xes_path)
    df  = pm4py.convert_to_dataframe(log)  # one row per event
    fname    = os.path.splitext(os.path.basename(xes_path))[0] + ".csv"
    csv_path = os.path.join(LABELED_DIR, fname)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

# 2) Run IMPresseD encoding
imp_nom.main()
