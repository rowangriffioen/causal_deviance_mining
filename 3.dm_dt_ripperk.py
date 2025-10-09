#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader

# ML components
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, _tree

import wittgenstein as lw  # RIPPER


def main():
    # Load configuration
    config_path = os.path.join("config", "config_dm_dt_ripperk.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    event_log  = config["event_log"]
    experiment = config["experiment_name"]
    encoding_cfg = config.get("encoding", None)
    max_k       = config["select_k"]
    folds       = config["folds"]
    random_state = config["random_state"]
    classifier   = config["classifier"]
    classifier_params_ripperk = config.get("classifier_params_ripperk", {})
    classifier_params_dt      = config.get("classifier_params_dt", {})
    target = config["target"]

    print(f"event log: {event_log}")
    print(f"experiment: {experiment}")
    print(f"encoding config: {encoding_cfg}")
    print(f"max features to keep (k): {max_k}")
    print(f"number of folds: {folds}")
    print(f"random state: {random_state}")
    print(f"classifier: {classifier}")

    # Determine encodings to process
    base_input = os.path.join("3.2_binned_features", event_log, f"{experiment}_features")
    if encoding_cfg:
        encodings = [encoding_cfg]
    else:
        encodings = [d for d in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, d))]

    # Process each encoding
    for enc in encodings:
        print(f"\n=== Deviance mining on encoding: {enc} ===")

        # Paths
        dataset_file = os.path.join(base_input, enc, f"{enc}.csv")
        out_dir = os.path.join("4_output", classifier, event_log, f"{experiment}_features", enc)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Dataset file: {dataset_file}")
        print(f"Output directory: {out_dir}")

        # Feature list from header (exclude case id and target)
        df_cols = pd.read_csv(dataset_file, nrows=0).columns.tolist()
        selected_vars = [c for c in df_cols if c not in ("Case_ID", target)]
        print(f"Selected variables: {selected_vars}")

        # Load data
        df = pd.read_csv(dataset_file)
        y = df[target]
        X = df[selected_vars]

        # Preprocessing pipelines
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ])

        # Feature selector + classifier
        selector = SelectKBest(mutual_info_classif, k=max_k)
        if classifier == "dt":
            clf = DecisionTreeClassifier(random_state=random_state, **classifier_params_dt)
        elif classifier == "ripperk":
            clf = lw.RIPPER(random_state=random_state, verbosity=0, **classifier_params_ripperk)
        else:
            raise ValueError("Unsupported classifier. Choose 'dt' or 'ripperk'.")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("feature_selection", selector),
            ("classifier", clf),
        ])

        # Cross-validation
        scoring = {
            "precision": "precision",
            "recall":    "recall",
            "f1":        "f1",
            "roc_auc":   "roc_auc",
        }
        cv_results = cross_validate(
            pipeline, X, y,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state),
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        df_cv = pd.DataFrame(cv_results).filter(like="test_")
        df_cv.columns = [c.replace("test_", "") for c in df_cv.columns]
        print(df_cv)
        print("Average:", df_cv.mean())

        # Fit on full dataset and time it
        start_time = time.time()
        pipeline.fit(X, y)
        elapsed = time.time() - start_time

        clf_full         = pipeline.named_steps["classifier"]
        preprocessor_full = pipeline.named_steps["preprocessor"]
        selector_full     = pipeline.named_steps["feature_selection"]

        # Recover feature names after preprocessing (handles no/zero categorical columns)
        num_cols = preprocessor_full.transformers_[0][2]
        feat_names = list(num_cols)

        cat_cols = preprocessor_full.transformers_[1][2]
        cat_tf   = preprocessor_full.named_transformers_.get("cat", None)

        ohe_names = []
        if cat_tf is not None and len(cat_cols) > 0:
            ohe = getattr(cat_tf, "named_steps", {}).get("ohe", cat_tf)
            if hasattr(ohe, "get_feature_names_out"):
                ohe_names = list(ohe.get_feature_names_out(cat_cols))

        feat_names.extend(ohe_names)

        sel_idx = selector_full.get_support(indices=True)
        selected_feat_names = [feat_names[i] for i in sel_idx]

        # Extract readable rules
        rules_list = []
        if classifier == "dt":
            tree = clf_full.tree_

            def recurse(node, conditions):
                feat_idx = tree.feature[node]
                if feat_idx != _tree.TREE_UNDEFINED:
                    name = selected_feat_names[feat_idx]
                    thresh = tree.threshold[node]
                    if np.isclose(thresh, 0.5):
                        left_cond  = f"{name} = 0"
                        right_cond = f"{name} = 1"
                    else:
                        left_cond  = f"{name} <= {thresh}"
                        right_cond = f"{name} > {thresh}"
                    recurse(tree.children_left[node],  conditions + [left_cond])
                    recurse(tree.children_right[node], conditions + [right_cond])
                else:
                    rules_list.append(f"[{' ∧ '.join(conditions)}] --> {target}")

            recurse(0, [])

        elif classifier == "ripperk":
            for rule in clf_full.ruleset_.rules:
                conds = []
                for cond in rule.conds:
                    idx = int(cond.feature)
                    fname = selected_feat_names[idx]
                    conds.append(f"{fname} = {cond.val}")
                rules_list.append(f"[{' ∧ '.join(conds) if conds else 'True'}] --> {target}")

        # Save rules + CV metrics
        avg_metrics = df_cv.mean()
        df_rules = pd.DataFrame({
            "rule":      rules_list,
            "precision": avg_metrics["precision"],
            "recall":    avg_metrics["recall"],
            "f1":        avg_metrics["f1"],
            "roc_auc":   avg_metrics["roc_auc"],
        })

        csv_name = f"rules_{classifier}_{enc}_{target}.csv"
        csv_path = os.path.join(out_dir, csv_name)
        df_rules.to_csv(csv_path, index=False)
        print(f"Saved {len(df_rules)} rules + metrics to:\n  {csv_path}")

        time_name = f"seconds_runtime_{classifier}_{experiment}_{enc}.txt"
        time_path = os.path.join(out_dir, time_name)
        with open(time_path, "w") as tf:
            tf.write(str(elapsed))
        print(f"Saved runtime ({elapsed:.2f}s) to:\n  {time_path}")


if __name__ == "__main__":
    main()
