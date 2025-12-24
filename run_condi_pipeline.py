import argparse
import os
import re
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="CoPHo conditional control pipeline")

    parser.add_argument(
        "--homo_condition_phase_s",
        type=float,
        default=0.6,
        help="PH_introduce_time: when to introduce persistent homology (default: 0.6)",
    )
    parser.add_argument(
        "--homo_condition_phase_e",
        type=float,
        default=1.0,
        help="When to stop condition control (default: 1.0)",
    )
    parser.add_argument(
        "--condition_target",
        type=str,
        choices=["clustering", "assortativity", "transitivity", "density"],
        required=True,
        help="Condition control target: one of {clustering, assortativity, transitivity, density}",
    )

    return parser.parse_args()


def clear_processed_dir():
    """
    Delete all files under data/comm20/processed/ (recursively),
    but keep all directories.
    """
    base_dir = Path("data/comm20/processed")
    if not base_dir.exists():
        print(f"[WARN] Directory does not exist: {base_dir}, skip clearing.")
        return

    removed_files = 0
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            fpath = Path(root) / fname
            try:
                fpath.unlink()
                removed_files += 1
            except Exception as e:
                print(f"[ERROR] Failed to delete file: {fpath}, error: {e}")

    print(f"[INFO] Deleted {removed_files} files (kept all directories).")


def update_condi_config(homo_s: float, homo_e: float, cond_target: str):
    """
    Update src/models/condi_config.py with given arguments:
      - homo_condition_phase_s
      - homo_condition_phase_e
      - condition_target = ["xxx"]
    """
    config_path = Path("src/models/condi_config.py")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")

    # Update homo_condition_phase_s
    text, n_s = re.subn(
        r"homo_condition_phase_s\s*=\s*[-+0-9.eE]+",
        f"homo_condition_phase_s = {homo_s}",
        text,
    )

    # Update homo_condition_phase_e
    text, n_e = re.subn(
        r"homo_condition_phase_e\s*=\s*[-+0-9.eE]+",
        f"homo_condition_phase_e = {homo_e}",
        text,
    )

    # Update condition_target (single-element list)
    text, n_target = re.subn(
        r'condition_target\s*=\s*\[[^\]]*\]',
        f'condition_target = ["{cond_target}"]',
        text,
    )

    # Simple sanity check
    if n_s == 0 or n_e == 0 or n_target == 0:
        print("[WARN] Some fields were not replaced in config. "
              "Please check the regex or if the template changed:")
        print(f"  homo_condition_phase_s replacements: {n_s}")
        print(f"  homo_condition_phase_e replacements: {n_e}")
        print(f"  condition_target replacements:       {n_target}")

    config_path.write_text(text, encoding="utf-8")
    print(
        f"[INFO] Updated {config_path}:\n"
        f"  homo_condition_phase_s = {homo_s}\n"
        f"  homo_condition_phase_e = {homo_e}\n"
        f'  condition_target       = ["{cond_target}"]'
    )


def run_main():
    """
    Run: python src/main.py
    """
    cmd = ["python", "src/main.py"]
    print(f"[INFO] Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"src/main.py failed, returncode = {result.returncode}")
    print("[INFO] src/main.py finished successfully.")


def main():
    args = parse_args()

    print("[STEP 1] Conditional data preparation: files under data/comm20/processed/ ")
    clear_processed_dir()

    print("[STEP 2] Update src/models/condi_config.py")
    update_condi_config(
        homo_s=args.homo_condition_phase_s,
        homo_e=args.homo_condition_phase_e,
        cond_target=args.condition_target,
    )

    print("[STEP 3] Run python src/main.py")
    run_main()


if __name__ == "__main__":
    main()