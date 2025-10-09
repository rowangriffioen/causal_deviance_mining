import os
from functools import reduce

from helper_functions.preparation.ConfigurationFile                 import ConfigurationFile
from helper_functions.preparation.PayloadType                       import PayloadType
from helper_functions.preparation.LogTaggingViaPredicates           import (
    tagLogWithOccurrence,
    tagLogWithValueEqOverEventAttn,
    tagLogWithSatAllProp,
    SatCases
)
from helper_functions.preparation.declaretemplates_new              import template_response
from helper_functions.feature_extraction.DumpUtils                  import read_single_arff_dump
from helper_functions.feature_extraction.PandaExpress               import ensureDataFrameQuality, fast_csv_parse


def build_conf(conf: dict) -> ConfigurationFile:
    """
    Build a ConfigurationFile using settings from a YAML-derived dict.

    Parameters:
    - exp_name: name of the experiment
    - unique_log_path: file path to the retagged XES log
    - conf: dict with keys:
        - output_folder: str
        - max_depth: int
        - min_leaf: int
        - sequence_threshold: int
        - payload_type: str (one of PayloadType enum names)
        - auto_ignore: List[str]
        - payload_settings: str (filename of payload rules)
    """
    cf = ConfigurationFile()
    cf.setExperimentName('experiment')
    #cf.setLogName(os.path.basename(unique_log_path))
    cf.setLogName(os.path.basename("log_name"))
    cf.setOutputFolder(conf.get("output_folder"))
    cf.setMaxDepth(conf.get("max_depth"))
    cf.setMinLeaf(conf.get("min_leaf"))
    cf.setSequenceThreshold(conf.get("sequence_threshold"))
    # Convert string to PayloadType enum member
    cf.setPayloadType(PayloadType[conf.get("payload_type")])
    cf.setAutoIgnore(conf.get("auto_ignore", []))
    cf.setPayloadSettings(conf.get("payload_settings"))
    return cf

def _make_value_eq(attribute, value):
    return lambda log: tagLogWithValueEqOverEventAttn(log, attribute, value)

def _make_occurrence(sequence, count):
    return lambda log: tagLogWithOccurrence(log, sequence, count)

def _make_sat_all_prop(predicates, sat_case):
    pred_list = []
    for p in predicates:
        tmpl = template_response if p.get("template") == "template_response" else p["template"]
        events = p.get("events", [])
        pred_list.append((tmpl, events))
    return lambda log: tagLogWithSatAllProp(log, pred_list, SatCases[sat_case])

# Load train/test dumps for one sequence encoding into pandas DataFrames
def read_sequence_log_via_arff(base_folder, encoding):
    """
    Load train and test encodings for a given sequence strategy.
    Returns two DataFrames (train_df, test_df), both indexed by Case_ID with numeric labels.
    """
    # Build the folder path for this encoding
    folder     = os.path.join(base_folder, encoding)
    arff_tr    = os.path.join(folder, "train_encodings.arff")
    arff_te    = os.path.join(folder, "test_encodings.arff")
    csv_tr     = os.path.join(folder, "crosstrain.csv")
    csv_te     = os.path.join(folder, "crosstest.csv")

    # Read and quality-check the ARFF + CSV pair for training and testing
    df_tr = read_single_arff_dump(arff_tr, csv_tr, doQualityCheck=True)
    df_te = read_single_arff_dump(arff_te, csv_te, doQualityCheck=True)

    # Ensure each DataFrame meets expected standards before returning
    return ensureDataFrameQuality(df_tr), ensureDataFrameQuality(df_te)

def internal_to_folder(folder_to_internal: dict) -> dict:
    """
    Build reverse mapping: from internal key → folder name
    """
    return {v: k for k, v in folder_to_internal.items()}


def read_all_numeric_columns_except_one(complete_path: str):
    """
    Read a numeric CSV with fast parser, printing the path for traceability
    """
    print(f"Reading: {complete_path}")
    return fast_csv_parse(complete_path)


def read_feature_csv(base_folder: str, encoding: str, folder_to_internal_map: dict):
    """
    Read the CSV for a given feature encoding and clean up.
    """
    folder = folder_to_internal_map[encoding]
    csv_path = os.path.join(base_folder, folder, f"{encoding}.csv")
    print(f"Reading feature file: {csv_path}")
    df = read_all_numeric_columns_except_one(csv_path)
    return df.drop(['index'], axis=1, errors='ignore')


def dataframe_join_with_checks(left, right):
    """
    Join two DataFrames on their index, with integrity checks on Case_ID and Label columns.
    """
    j = left.join(right, lsuffix='_left', rsuffix='_right')

    # If Case_ID columns present, verify alignment
    if 'Case_ID_left' in j.columns and 'Case_ID_right' in j.columns:
        assert j['Case_ID_left'].astype(str).tolist() == j['Case_ID_right'].astype(str).tolist(), \
            "Case_ID mismatch between left and right"
        j.drop('Case_ID_left', axis=1, inplace=True)
        j.rename(columns={'Case_ID_right': 'Case_ID'}, inplace=True)

    # Verify that labels match
    if 'Label_left' in j.columns and 'Label_right' in j.columns:
        assert j['Label_left'].astype(int).tolist() == j['Label_right'].astype(int).tolist(), \
            "Label mismatch between left and right"
        j.drop('Label_left', axis=1, inplace=True)
        j.rename(columns={'Label_right': 'Label'}, inplace=True)

    return j


def dataframe_multiway_equijoin(dfs: list) -> any:
    """
    Multiway equi-join for a list of DataFrames using dataframe_join_with_checks
    """
    return reduce(dataframe_join_with_checks, dfs)


def multijoined_dump_no_splits(
    key: str,
    dataset_list: list,
    base_folder: str,
    folder_to_internal_map: dict,
    output_folder: str = None
) -> any:
    """
    Read each encoding in dataset_list, join them, and dump to CSV under its own subfolder.
    """
    # Read and collect all DataFrames
    dfs = [read_feature_csv(base_folder, enc, folder_to_internal_map) for enc in dataset_list]
    joined = dataframe_multiway_equijoin(dfs)

    # Determine and create output folder: either provided or a subfolder named after key
    out_folder = output_folder if output_folder else os.path.join(base_folder, key)
    os.makedirs(out_folder, exist_ok=True)

    # Write the joined CSV into its own folder
    out_path = os.path.join(out_folder, f"{key}.csv")
    joined.to_csv(out_path, index=False)
    print(f"Wrote joined features for '{key}' → {out_path}")
    return joined


def dump_all_compositions(
    dataset_composition: dict,
    base_folder: str,
    folder_to_internal_map: dict
) -> None:
    """
    Loop through compositions and call multijoined_dump_no_splits for each.
    """
    for key, encodings in dataset_composition.items():
        print(f"Processing composition: {key}")
        multijoined_dump_no_splits(key, encodings, base_folder, folder_to_internal_map)