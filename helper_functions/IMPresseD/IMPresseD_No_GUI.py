#!/usr/bin/env python3
"""
IMPresseD Improved CLI (no GUI) version with auto discovery
Includes enhancements: support for numerical and categorical attributes,
pairwise case-distance caching, and automatic pattern discovery using AutoStepWise_PPD.
"""
import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
from paretoset import paretoset

from .Auto_IMPID import AutoStepWise_PPD
from .IMIPD import (
    VariantSelection,
    create_pattern_attributes,
    calculate_pairwise_case_distance
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="IMPresseD improved auto mode without GUI"
    )
    parser.add_argument(
        '--log_path', required=True, type=str,
        help='Path to input CSV log'
    )
    parser.add_argument(
        '--output_path', required=True, type=str,
        help='Directory to save output files'
    )
    parser.add_argument(
        '--discovery_type', default='auto', choices=['auto'], type=str,
        help='Discovery mode (only auto supported)'
    )
    parser.add_argument(
        '--encoding', default=False, type=bool,
        help='(Ignored) Encoding flag from legacy script'
    )
    parser.add_argument(
        '--case_id', required=True, type=str,
        help='Case ID column name'
    )
    parser.add_argument(
        '--activity', required=True, type=str,
        help='Activity column name'
    )
    parser.add_argument(
        '--timestamp', required=True, type=str,
        help='Timestamp column name'
    )
    parser.add_argument(
        '--outcome', required=True, type=str,
        help='Outcome column name'
    )
    parser.add_argument(
        '--outcome_type', default='binary', choices=['binary', 'numerical'], type=str,
        help='Outcome type'
    )
    parser.add_argument(
        '--delta_time', default=0.0, type=float,
        help='Delta time in seconds'
    )
    parser.add_argument(
        '--max_gap', default=5.0, type=float,
        help='Maximum gap between events for eventual relations'
    )
    parser.add_argument(
        '--max_extension_step', default=2, type=int,
        help='Maximum number of extension steps'
    )
    parser.add_argument(
        '--testing_percentage', default=0.2, type=float,
        help='Testing data percentage'
    )
    parser.add_argument(
        '--likelihood', default='likelihood', type=str,
        help='(Ignored) Likelihood column name from legacy script'
    )
    parser.add_argument(
        '--extension_style', default='All', type=str,
        help='(Ignored) Extension style from legacy script'
    )
    parser.add_argument(
        '--numerical_attributes', nargs='*', default=[],
        help='List of numerical attribute column names'
    )
    parser.add_argument(
        '--categorical_attributes', nargs='*', default=[],
        help='List of categorical attribute column names'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Load event log
    df = pd.read_csv(args.log_path)
    # Basic preprocessing
    df[args.activity] = df[args.activity].astype(str).str.replace("_", "-")
    df[args.timestamp] = pd.to_datetime(df[args.timestamp])
    df[args.case_id] = df[args.case_id].astype(str)

    # Assign colors to activities
    unique_activities = df[args.activity].unique()
    color_codes = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                   for _ in unique_activities]
    color_act_dict = {act: code for act, code in zip(unique_activities, color_codes)}
    color_act_dict['start'] = 'k'
    color_act_dict['end'] = 'k'

    # Create patient-level features
    patient_data = pd.DataFrame()
    if args.numerical_attributes:
        patient_data[args.numerical_attributes] = df[args.numerical_attributes]
    if args.categorical_attributes:
        patient_data[args.categorical_attributes] = df[args.categorical_attributes]
    patient_data[args.case_id] = df[args.case_id]
    patient_data[args.outcome] = df[args.outcome]
    patient_data = patient_data.drop_duplicates(subset=[args.case_id], keep='first')
    patient_data = patient_data.sort_values(by=args.case_id).reset_index(drop=True)

    # Initialize activity count columns
    for act in unique_activities:
        patient_data[act] = 0

    # Variant-based feature enrichment
    variants = VariantSelection(df, args.case_id, args.activity, args.timestamp)
    for case in variants['case:concept:name'].unique():
        other_cases = variants.loc[
            variants['case:concept:name'] == case, 'case:CaseIDs']
        if not other_cases.empty:
            other_list = other_cases.tolist()[0]
            trace = df.loc[df[args.case_id] == case, args.activity].tolist()
            for act in np.unique(trace):
                count_act = trace.count(act)
                for ocase in other_list:
                    patient_data.loc[
                        patient_data[args.case_id] == ocase, act
                    ] = count_act

    # Compute or load pairwise case distances
    log_dir = os.path.dirname(os.path.abspath(args.log_path))
    dist_dir = os.path.join(log_dir, "dist")
    os.makedirs(dist_dir, exist_ok=True)
    dist_file = os.path.join(dist_dir, "pairwise_case_distances.pkl")
    if os.path.exists(dist_file):
        with open(dist_file, 'rb') as f:
            pairwise_distances_array = pickle.load(f)
    else:
        X_features = patient_data.drop([args.case_id, args.outcome], axis=1)
        pairwise_distances_array = calculate_pairwise_case_distance(
            X_features, args.numerical_attributes
        )
        with open(dist_file, 'wb') as f:
            pickle.dump(pairwise_distances_array, f)

    # Build pair_cases and start_search_points
    case_indices = list(patient_data.index)
    pair_cases = [
        (a, b)
        for idx, a in enumerate(case_indices)
        for b in case_indices[idx + 1:]
    ]
    n_cases = len(case_indices)
    start_search_points = []
    i = 0
    for k in range(n_cases):
        start_search_points.append(k * n_cases - (i + k))
        i += k

    # Automatic pattern discovery
    pareto_features = [
        'Outcome_Interest', 'Frequency_Interest', 'Case_Distance_Interest'
    ]
    pareto_sense = ['max', 'max', 'min']

    train_X, test_X = AutoStepWise_PPD(
        args.max_extension_step,
        args.max_gap,
        args.testing_percentage,
        df,
        patient_data,
        pairwise_distances_array,
        pair_cases,
        start_search_points,
        args.case_id,
        args.activity,
        args.outcome,
        args.outcome_type,
        args.timestamp,
        pareto_features,
        pareto_sense,
        args.delta_time,
        color_act_dict,
        args.output_path
    )

    # Save encoded logs
    train_X.to_csv(
        os.path.join(args.output_path, "training_encoded_log.csv"),
        index=False
    )
    test_X.to_csv(
        os.path.join(args.output_path, "testing_encoded_log.csv"),
        index=False
    )

    print("Auto pattern discovery completed. Results saved to", args.output_path)


if __name__ == '__main__':
    main()
