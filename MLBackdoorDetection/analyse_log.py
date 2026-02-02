#!/usr/bin/env python3
import re
import csv
import argparse
from pathlib import Path

# Regex: absolute path to .pth + two floats
LINE_RE = re.compile(
    r'(/[^ \n]+\.pth)\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)'
)

def parse_log_file(filepath):
    """
    Parse a single log file and return a list of tuples:
    (model_path, anomaly_metric, time_cost)
    """
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = LINE_RE.search(line)
                if m:
                    model_path = m.group(1)
                    anomaly_metric = m.group(2)
                    time_cost = m.group(3)
                    results.append((model_path, anomaly_metric, time_cost))
    except Exception as e:
        print(f"[WARN] Failed to parse {filepath}: {e}")
    return results

def main():
    ap = argparse.ArgumentParser(
        description="Parse batch logs to CSV (filepath, anomaly metric, time cost)."
    )
    ap.add_argument(
        "logs",
        nargs="+",
        help="One or more log files (supports shell glob expansion, e.g., batch_*.log)"
    )
    ap.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV filepath"
    )
    args = ap.parse_args()

    all_rows = []
    for log in args.logs:
        p = Path(log)
        if not p.exists():
            print(f"[WARN] Log file not found: {log}")
            continue
        rows = parse_log_file(str(p))
        if not rows:
            print(f"[INFO] No matching lines found in {log}")
        else:
            all_rows.extend(rows)

    if not all_rows:
        print("[INFO] No data extracted; CSV will not be created.")
        return

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath", "anomaly metric", "time cost"])
        writer.writerows(all_rows)

    print(f"[OK] Wrote {len(all_rows)} rows to {out_path}")

if __name__ == "__main__":
    main()