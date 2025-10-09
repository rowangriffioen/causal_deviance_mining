"""
Code to find and encode IA encoding (Individual Activities) or baseline

"""
from helper_functions.feature_extraction.deviancecommon import extract_unique_events_transformed
import numpy as np
import pandas as pd
from helper_functions.feature_extraction.PathUtils import *
from helper_functions.feature_extraction.DumpUtils import genericDump, dump_in_primary_memory_as_table_csv


def transform_log(train_log, activity_set):
    train_names = []

    train_labels = []
    train_data = []
    for trace in train_log:
        name = trace["name"]
        label = trace["label"]
        res = []
        train_labels.append(label)
        train_names.append(name)
        for event in activity_set:
            if event in trace["events"]:
                res.append(len(trace["events"][event]))
            else:
                res.append(0)

        train_data.append(res)

    np_train_data = np.array(train_data)
    train_df = pd.DataFrame(np_train_data)
    train_df.columns = activity_set
    train_df["Case_ID"] = train_names
    train_df["Label"] = train_labels
    train_df.set_index("Case_ID")
    return train_df

def baseline_embedding(inp_folder, log, self=None):
    """
    Baseline embedding (Individual Activities) from a propositional log.
    
    Parameters:
    - inp_folder: Output folder path
    - log: A list of propositionalized traces (each as a dict with 'name', 'events', 'label')
    - self: Optional ExperimentRunner instance (can be None)
    
    Returns:
    - Set of trace (case) IDs written to the baseline.csv
    """
    activitySet = list(extract_unique_events_transformed(log))

    # Transform to feature matrix
    if len(log) > 0:
        df = transform_log(log, activitySet)
    else:
        df = pd.DataFrame()

    mkdir_test(inp_folder)
    test_df = df.iloc[0:0]                # empty DataFrame for “test” split
    genericDump(inp_folder, df, test_df, "baseline.csv", None)

    if self is not None:
        dump_in_primary_memory_as_table_csv(self, "bs", df, None)

    return set(df["Case_ID"]) if "Case_ID" in df else set()





