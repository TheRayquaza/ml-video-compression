#!/usr/bin/env python3
"""Aggregate PSNR from results_*_epochs.csv files and plot PSNR vs epochs.

Outputs PNG files into ./plots/:
- per-video: <video_basename>_psnr_vs_epochs.png
- average: average_psnr_vs_epochs.png

Requires: pandas, matplotlib
"""
import sys
import re
import glob
import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def parse_epochs_from_filename(path):
    m = re.search(r"results_(\d+)_epochs_2_scale\.csv$", path)
    if m:
        return int(m.group(1))
    return None


def find_result_csvs(root_dir="."):
    pattern = os.path.join(root_dir, "results_*.csv")
    print(pattern)
    return sorted(glob.glob(pattern))


def load_all(csv_paths):
    data = {}
    for p in csv_paths:
        epochs = parse_epochs_from_filename(p)
        print(epochs)
        if epochs is None:
            continue
        df = pd.read_csv(p)
        # Expect column mean_psnr_db and filename
        if "mean_psnr_db" not in df.columns or "filename" not in df.columns:
            print(f"Skipping {p}: missing expected columns")
            continue
        data[epochs] = df
    return data


def aggregate_by_video(data_by_epochs):
    # data_by_epochs: {epochs: DataFrame}
    videos = defaultdict(dict)
    for epochs, df in data_by_epochs.items():
        for _, row in df.iterrows():
            print(row)
            vid = row["filename"]
            psnr = float(row["mean_psnr_db"]) if not pd.isna(row["mean_psnr_db"]) else None
            videos[vid][epochs] = psnr
    return videos


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_all_videos_single_plot(videos_dict, out_dir="plots"):
    """Plot a single figure with one line per video and an average line."""
    ensure_dir(out_dir)
    all_epochs = sorted({e for v in videos_dict.values() for e in v.keys()})

    plt.figure(figsize=(10, 6))
    # plot each video
    for vid, epoch_map in sorted(videos_dict.items()):
        epochs = sorted(epoch_map.keys())
        psnrs = [epoch_map[e] for e in epochs]
        label = os.path.splitext(os.path.basename(vid))[0]
        plt.plot(epochs, psnrs, marker='o', label=label)

    # compute and plot average
    avg_x = []
    avg_y = []
    for e in all_epochs:
        vals = [v[e] for v in videos_dict.values() if e in v and v[e] is not None]
        if vals:
            avg_x.append(e)
            avg_y.append(sum(vals) / len(vals))

    if avg_x:
        plt.plot(avg_x, avg_y, linestyle='--', color='k', linewidth=2.0, label='average')

    plt.title("PSNR vs epochs with Scaling of 2 — sources/ videos")
    plt.xlabel("epochs")
    plt.ylabel("mean PSNR (dB)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize='small')

    outpath = os.path.join(out_dir, "psnr_all_videos_vs_epochs_scale_4.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"Wrote {outpath}")


def main():
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = os.path.dirname(__file__)
    print(root)
    csvs = find_result_csvs(root)
    if not csvs:
        print("No result CSVs found (looking for results_*_epochs.csv)")
        return

    print("Found CSVs:")
    for c in csvs:
        print(" -", os.path.basename(c))

    data = load_all(csvs)
    videos = aggregate_by_video(data)
    print(f"Found {len(videos)} unique video entries")
    plot_all_videos_single_plot(videos, out_dir=os.path.join(root, "plots"))


if __name__ == "__main__":
    main()
