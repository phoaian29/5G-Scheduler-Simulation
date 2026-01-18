import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALT_KEYS = {
    "ep_rew_mean": ["rollout/ep_rew_mean", "ep_rew_mean"],
    "value_loss": ["train/value_loss", "value_loss"],
    "entropy_loss": ["train/entropy_loss", "entropy_loss"],
    "Jain Fairness": ["custom/jain_index", "kpi/jain_mean", "jain"],
    "Cell Throughput (Mbps)": ["custom/cell_tput_Mb", "kpi/cell_tput_Mb_per_tti", "cell_tput_Mb"],
}

def moving_avg(a, k=10):
    if len(a) < k: return a
    return np.convolve(a, np.ones(k), 'valid') / k

def safe_plot(x, y, title, xlabel, ylabel, out_png):
    if len(x) == 0 or len(y) == 0: 
        print(f"[WARNING] No data to plot for {title}")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=1.5, alpha=0.4, label="Raw", color='gray')
    if len(y) > 10:
        ma_y = moving_avg(y)
        ma_x = x[len(x)-len(ma_y):]
        plt.plot(ma_x, ma_y, linewidth=2.0, label="Trend", color='blue')
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"[SAVE] Plot saved: {out_png}")

def visualize_progress(logdir, outdir):
    progress_csv = os.path.join(logdir, "progress.csv")
    if not os.path.exists(progress_csv):
        print(f"[ERROR] Not found: {progress_csv}")
        return
    
    try:
        df = pd.read_csv(progress_csv)
    except pd.errors.EmptyDataError:
        print(f"[ERROR] CSV file is empty: {progress_csv}")
        print("[INFO] Training may have just started. Run again after more steps complete.")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return
    
    if df.empty or len(df) == 0:
        print(f"[WARNING] CSV file has no rows: {progress_csv}")
        print("[INFO] Training may have just started. Run again after more steps complete.")
        return
    
    x = df.index.values

    plots = [
        ("ep_rew_mean", "Reward Mean"),
        ("value_loss", "Value Loss"),
        ("entropy_loss", "Entropy"),
        ("Jain Fairness", "Jain Fairness"),
        ("Cell Throughput (Mbps)", "Throughput (Mbps)")
    ]

    for key, title in plots:
        for col in ALT_KEYS[key]:
            if col in df.columns:
                safe_plot(x, df[col].values, title, "Steps", key, 
                          os.path.join(outdir, f"{key.split()[0]}.png"))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/matlab_ppo_logging")
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[INFO] Reading logs from: {args.logdir}")
    print(f"[INFO] Saving plots to: {args.outdir}")
    visualize_progress(args.logdir, args.outdir)
    print("[SUCCESS] Visualization complete!")