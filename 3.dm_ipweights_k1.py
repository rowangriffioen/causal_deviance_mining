#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import yaml
from yaml.loader import SafeLoader

BASE_DIR = Path(__file__).resolve().parent  # folder of this script

def main():
    # Load base configuration
    config_path = BASE_DIR / "config" / "config_dm_crm_k1.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)

    event_log    = config["event_log"]
    experiment   = config["experiment_name"]
    encoding_cfg = config.get("encoding")

    print("CWD:", Path.cwd())
    print("BASE_DIR:", BASE_DIR)
    print("event log:", event_log)
    print("experiment:", experiment)
    print("encoding config:", encoding_cfg)

    # Determine encodings to process
    base_input = BASE_DIR / "3.2_binned_features" / event_log / f"{experiment}_features"
    if encoding_cfg:
        encodings = [encoding_cfg]
    else:
        encodings = [d.name for d in base_input.iterdir() if d.is_dir()]

    # Fixed settings from config
    target_var = config["target"]

    for enc in encodings:
        print(f"\n=== CRM (IP-weights) on encoding: {enc} ===")

        # Paths
        dataset_file_abs = (base_input / enc / f"{enc}.csv").resolve()
        out_dir = BASE_DIR / "4_output" / "ipweights" / event_log / f"{experiment}_features" / "k1" / enc
        out_dir.mkdir(parents=True, exist_ok=True)

        if not dataset_file_abs.exists():
            print(f" ⚠️  Missing dataset file: {dataset_file_abs}")
            continue

        # Read header to determine feature columns (absolute path)
        df_cols = pd.read_csv(dataset_file_abs, nrows=0).columns.tolist()
        selected_vars = [c for c in df_cols if c not in ("Case_ID", "Label")]

        # Write RELATIVE path into YAML (avoids './C:\\...' issues in called script)
        try:
            dataset_file_rel = dataset_file_abs.relative_to(BASE_DIR)
        except ValueError:
            dataset_file_rel = dataset_file_abs  # fallback if outside BASE_DIR

        crm_cfg = {
            "event_log": event_log,
            "experiment_name": experiment,
            "encoding": enc,
            "dataset_file": str(dataset_file_rel).replace("\\", "/"),
            "name": f"{enc}_k1",
            "selected variables": selected_vars,
            "controllable variables": selected_vars.copy(),
            "nominal variables": [],          # none by default
            "target": target_var,
        }
        tmp_cfg = BASE_DIR / f"temp_crm_ipweights_{experiment}_{enc}_k1.yaml"
        tmp_cfg.write_text(yaml.dump(crm_cfg), encoding="utf-8")

        # Execute discovery script; on success move artifacts; always clean temp config
        try:
            subprocess.run(
                [sys.executable, "models/CRM_ipweights_k1.py", tmp_cfg.name],
                check=True,
                cwd=BASE_DIR,  # important for resolving relative paths inside the model script
            )

            generated = f"rules_{enc}_k1_{target_var}.csv"
            timing    = f"seconds_runtime_{experiment}_{enc}_k1.txt"

            os.replace(BASE_DIR / generated, out_dir / generated)
            os.replace(BASE_DIR / timing,    out_dir / timing)

            print(f" → Causal-rule CSV moved to: {out_dir / generated}")
            print(f" → Time TXT moved to: {out_dir / timing}")

        except subprocess.CalledProcessError as e:
            print(f" ⚠️  CRM script failed for encoding {enc} (exit code {e.returncode}). Skipping.")
        except FileNotFoundError as e:
            print(f" ⚠️  File operation failed for encoding {enc}: {e}. Skipping.")
        finally:
            if tmp_cfg.exists():
                tmp_cfg.unlink()

if __name__ == "__main__":
    main()
