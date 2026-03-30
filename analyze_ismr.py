import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def parse_svid(svid_str):
    """Parses SVID string which can be a single ID, a comma-separated list, or a range."""
    svids = set()
    if not svid_str:
        return None
    
    parts = svid_str.split(',')
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                svids.update(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range format '{part}'. Skipping.")
        else:
            try:
                svids.add(int(part))
            except ValueError:
                print(f"Warning: Invalid SVID format '{part}'. Skipping.")
    return list(svids)

def main():
    parser = argparse.ArgumentParser(description="Analyze ISMR CSV data.")
    parser.add_argument("--input", required=True, help="Path to the target ISMR file.")
    parser.add_argument("--svid", help="SVID(s) to filter (e.g., '15', '1,5,12', or '10-20').")
    parser.add_argument("--elev-min", type=float, help="Minimum elevation filter.")
    parser.add_argument("--elev-max", type=float, help="Maximum elevation filter.")
    parser.add_argument("--x-cols", nargs='+', required=True, help="Column names to plot on the x-axis.")
    parser.add_argument("--y-col", required=True, help="Column name to plot on the y-axis.")
    parser.add_argument("--gap-threshold", type=float, default=60.0, help="Threshold for TOW discontinuity in seconds (default: 60).")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Starting analysis on {args.input}...")

if __name__ == "__main__":
    main()
