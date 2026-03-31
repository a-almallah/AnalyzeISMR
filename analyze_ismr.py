import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def parse_svid(svid_str):
    """Parses SVID string which can be a single ID, a comma-separated list, or a range."""
    if not svid_str:
        return None
    svids = set()
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

def are_similar(y1, y2):
    """Determines if two parameters are similar enough to share a Y-axis scale."""
    groups = [
        ["S4", "S4_corr"],
        ["Phi01", "Phi03", "Phi10", "Phi30", "Phi60"],
        ["TEC", "dTEC"],
        ["CN0"],
        ["LockTime"],
        ["SI"],
        ["AvgCCD", "SigmaCCD"],
        ["Azimuth", "Elevation", "T"]
    ]
    for group in groups:
        match1 = any(g in y1 for g in group)
        match2 = any(g in y2 for g in group)
        if match1 and match2:
            return True
    return False

def load_and_filter(args):
    """Loads ISMR data, applies filters, and handles missing columns."""
    header = [
        "WN", "TOW", "SVID", "RxState", "Azimuth", "Elevation", "Sig1_CN0", "Sig1_S4", "Sig1_S4_corr", 
        "Sig1_Phi01", "Sig1_Phi03", "Sig1_Phi10", "Sig1_Phi30", "Sig1_Phi60", "Sig1_AvgCCD", "Sig1_SigmaCCD", 
        "TEC_45", "dTEC_60_45", "TEC_30", "dTEC_45_30", "TEC_15", "dTEC_30_15", "TEC_0", "dTEC_15_0", 
        "Sig1_LockTime", "sbf2ismr_version", "Sig2_LockTime_TEC", "Sig2_CN0_TEC", "Sig1_SI", "Sig1_SI_num", 
        "Sig1_p", "Sig2_CN0", "Sig2_S4", "Sig2_S4_corr", "Sig2_Phi01", "Sig2_Phi03", "Sig2_Phi10", 
        "Sig2_Phi30", "Sig2_Phi60", "Sig2_AvgCCD", "Sig2_SigmaCCD", "Sig2_LockTime", "Sig2_SI", 
        "Sig2_SI_num", "Sig2_p", "Sig3_CN0", "Sig3_S4", "Sig3_S4_corr", "Sig3_Phi01", "Sig3_Phi03", 
        "Sig3_Phi10", "Sig3_Phi30", "Sig3_Phi60", "Sig3_AvgCCD", "Sig3_SigmaCCD", "Sig3_LockTime", 
        "Sig3_SI", "Sig3_SI_num", "Sig3_p", "Sig1_T", "Sig2_T", "Sig3_T"
    ]

    try:
        df = pd.read_csv(args.input, names=header, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    required_cols = ["TOW", "SVID", "Elevation"] + args.x_cols + args.y_cols
    lock_time_cols = [c for c in header if "LockTime" in c]
    required_cols.extend(lock_time_cols)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        user_specified = set(args.x_cols + args.y_cols + ["TOW", "SVID", "Elevation"])
        missing_user = [c for c in missing if c in user_specified]
        if missing_user:
            print(f"Error: Missing columns in data: {missing_user}")
            sys.exit(1)

    # Normalize TOW to fractional hours
    df['TOW_HOURS'] = df['TOW'] / 3600.0

    svids = parse_svid(args.svid)
    if svids:
        df = df[df["SVID"].isin(svids)]

    if args.elev_min is not None:
        df = df[df["Elevation"] >= args.elev_min]
    if args.elev_max is not None:
        df = df[df["Elevation"] <= args.elev_max]

    if df.empty:
        print("Warning: Dataframe is empty after filtering.")
    
    return df

def get_segments(df, gap_threshold):
    """
    Identifies continuous segments based on TOW gaps and LockTime resets.
    Returns a list of dataframes, each representing a continuous segment.
    """
    if df.empty:
        return []

    df = df.sort_values(by=["SVID", "TOW"]).copy()
    segments = []
    
    for svid in df["SVID"].unique():
        sv_df = df[df["SVID"] == svid].copy()
        tow_gap = sv_df["TOW"].diff() > gap_threshold
        
        lock_cols = [c for c in sv_df.columns if "LockTime" in c]
        lock_reset = pd.Series(False, index=sv_df.index)
        for col in lock_cols:
             lock_reset |= (sv_df[col].diff() < 0)

        discontinuity = tow_gap | lock_reset
        sv_df["segment_id"] = discontinuity.cumsum()
        
        for _, segment in sv_df.groupby("segment_id"):
            segments.append(segment)
            
    return segments

def plot_data(segments, x_cols, y_cols, output_path=None, compress=False):
    """
    Plots segments with specified markers for discontinuities.
    If compress is True and 2 non-similar Ys, uses dual Y axes.
    """
    if not segments:
        print("No segments to plot.")
        return

    if output_path:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    if compress:
        num_plots = len(x_cols)
    else:
        num_plots = len(x_cols) * len(y_cols)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 6 * num_plots), 
                             sharex=(len(x_cols) == 1), constrained_layout=True)
    
    if num_plots == 1:
        axes = [axes]

    all_svids = sorted(list(set(seg["SVID"].iloc[0] for seg in segments)))
    try:
        color_map = plt.get_cmap("tab20")
    except AttributeError:
        color_map = plt.cm.get_cmap("tab20")
    
    svid_to_color = {svid: color_map(i % 20) for i, svid in enumerate(all_svids)}
    y_styles = ['-', '--']

    svid_segments = {}
    for seg in segments:
        svid = seg["SVID"].iloc[0]
        if svid not in svid_segments:
            svid_segments[svid] = []
        svid_segments[svid].append(seg)

    plot_idx = 0
    for x_col in x_cols:
        target_y_groups = [y_cols] if compress else [[y] for y in y_cols]
        
        for y_group in target_y_groups:
            ax1 = axes[plot_idx]
            
            # Check if we need dual axes
            use_twin = False
            if len(y_group) == 2 and not are_similar(y_group[0], y_group[1]):
                use_twin = True
                ax2 = ax1.twinx()
            
            for y_idx, y_col in enumerate(y_group):
                curr_ax = ax1 if (y_idx == 0 or not use_twin) else ax2
                style = y_styles[y_idx % 2]
                
                for svid, segs in svid_segments.items():
                    color = svid_to_color[svid]
                    for seg_idx, segment in enumerate(segs):
                        x_data = segment['TOW_HOURS'] if x_col == 'TOW' else segment[x_col]
                        
                        label = f"SVID {svid}"
                        if len(y_group) > 1:
                            label += f" ({y_col})"
                        
                        curr_ax.plot(x_data, segment[y_col], color=color, linestyle=style,
                                     label=label if plot_idx == 0 and seg_idx == 0 else "")
                        
                        # Discontinuity markers
                        if seg_idx > 0:
                            curr_ax.plot(x_data.iloc[0], segment[y_col].iloc[0], 
                                         marker='o', color=color, markersize=6)
                        if seg_idx < len(segs) - 1:
                            curr_ax.plot(x_data.iloc[-1], segment[y_col].iloc[-1], 
                                         marker='o', markerfacecolor='none', markeredgecolor=color, markersize=6)

            ax1.set_xlabel(x_col if x_col != 'TOW' else "TOW (Hours)")
            ax1.grid(True)

            if use_twin:
                ax1.set_ylabel(y_group[0], color='black')
                ax2.set_ylabel(y_group[1], color='black')
            else:
                ax1.set_ylabel(", ".join(y_group))
            
            # Legend handling
            if plot_idx == 0:
                h1, l1 = ax1.get_legend_handles_labels()
                if use_twin:
                    h2, l2 = ax2.get_legend_handles_labels()
                    h1.extend(h2)
                    l1.extend(l2)
                
                by_label = dict(zip(l1, h1))
                ax1.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=3, fontsize='small')
            
            plot_idx += 1

    plt.suptitle(f"ISMR Analysis: {', '.join(y_cols)} vs {', '.join(x_cols)}")
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze ISMR CSV data.")
    parser.add_argument("--input", required=True, help="Path to the target ISMR file.")
    parser.add_argument("--svid", help="SVID(s) to filter (e.g., '15', '1,5,12', or '10-20').")
    parser.add_argument("--elev-min", type=float, help="Minimum elevation filter.")
    parser.add_argument("--elev-max", type=float, help="Maximum elevation filter.")
    parser.add_argument("--x-cols", nargs='+', required=True, help="Column names to plot on the x-axis.")
    parser.add_argument("--y-cols", nargs='+', required=True, help="Column names to plot on the y-axis.")
    parser.add_argument("--gap-threshold", type=float, default=60.0, help="Threshold for TOW discontinuity in seconds (default: 60).")
    parser.add_argument("--output", help="Path to save the plot (e.g., 'plot.png'). If not provided, shows the plot.")
    parser.add_argument("--compress", action="store_true", help="Plot all Y columns on a single plot per X column.")

    args = parser.parse_args()

    if len(args.y_cols) > 2:
        print("Error: --y-cols is capped to a maximum of 2 parameters.")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Loading and filtering data from {args.input}...")
    df = load_and_filter(args)
    
    if df.empty:
        print("No data to plot.")
        return

    print("Identifying continuous segments...")
    segments = get_segments(df, args.gap_threshold)
    print(f"Found {len(segments)} segments across {len(df['SVID'].unique())} SVIDs.")

    print("Plotting data...")
    try:
        plot_data(segments, args.x_cols, args.y_cols, args.output, args.compress)
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
