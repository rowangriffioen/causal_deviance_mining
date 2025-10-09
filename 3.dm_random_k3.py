#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import yaml
from yaml.loader import SafeLoader


def main():
    # Load base configuration
    config_path = os.path.join("config", "config_dm_crm_k3.yaml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    event_log    = config["event_log"]
    experiment   = config["experiment_name"]
    encoding_cfg = config.get("encoding", None)

    print("event log:", event_log)
    print("experiment:", experiment)
    print("encoding config:", encoding_cfg)

    # Determine encodings to process
    base_input = os.path.join("3.2_binned_features", event_log, f"{experiment}_features")
    if encoding_cfg:
        encodings = [encoding_cfg]
    else:
        encodings = [d for d in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, d))]

    # Fixed settings from config
    target_var = config["target"]

    for enc in encodings:
        print(f"\n=== CRM on encoding: {enc} ===")

        # Paths
        dataset_file = os.path.join("3.2_binned_features", event_log, f"{experiment}_features", enc, f"{enc}.csv")
        out_dir = os.path.join("4_output", "random", event_log, f"{experiment}_features", "k3", enc)
        os.makedirs(out_dir, exist_ok=True)

        # Read header to determine feature columns
        df_cols = pd.read_csv(dataset_file, nrows=0).columns.tolist()
        selected_vars = [c for c in df_cols if c not in ("Case_ID", "Label")]

        # Build per-encoding CRM config
        crm_cfg = {
            "event_log": event_log,
            "experiment_name": experiment,
            "encoding": enc,
            "dataset_file": dataset_file,
            "name": f"{enc}_k3",
            "selected variables": selected_vars,
            "controllable variables": selected_vars.copy(),
            "nominal variables": [],          # none by default
            "target": target_var,
        }
        tmp_cfg = f"temp_crm_random_{experiment}_{enc}_k3.yaml"
        with open(tmp_cfg, "w") as tf:
            yaml.dump(crm_cfg, tf)

        # Execute discovery script; on success move artifacts; always clean temp config
        try:
            subprocess.run(["python", "models/CRM_random_k3.py", tmp_cfg], check=True)

            generated = f"rules_random_{enc}_k3_{target_var}.csv"
            timing    = f"seconds_runtime_random_{experiment}_{enc}_k3.txt"

            os.replace(generated, os.path.join(out_dir, generated))
            os.replace(timing,    os.path.join(out_dir, timing))

            print(f" → Causal-rule CSV moved to: {out_dir}/{generated}")
            print(f" → Time TXT moved to: {out_dir}/{timing}")

        except subprocess.CalledProcessError as e:
            print(f" ⚠️  CRM script failed for encoding {enc} (exit code {e.returncode}). Skipping.")
        except FileNotFoundError as e:
            print(f" ⚠️  File operation failed for encoding {enc}: {e}. Skipping.")
        finally:
            if os.path.exists(tmp_cfg):
                os.remove(tmp_cfg)


if __name__ == "__main__":
    main()
