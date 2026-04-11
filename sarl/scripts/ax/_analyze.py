#!/usr/bin/env python3
"""
Analyze and visualize grid search and Bayesian optimization results.

Usage:
    # Grid search analysis
    python analyze.py runs/2026-03-16_03-00
    python analyze.py runs/2026-03-16_03-00 runs/2026-03-16_05-00
    python analyze.py "runs/*/"

    # Bayesian optimization progress (live during run)
    python analyze.py --bo-progress runs/2026-03-23_02-12
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def extract_run_id(path: str) -> str:
    """Extract timestamp/run ID from path."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})', path)
    return match.group(1) if match else "unknown"


def load_grid_results(csv_path: Path, run_id: str) -> pd.DataFrame:
    """Load a single grid_results.csv file."""
    df = pd.read_csv(csv_path)
    df['run_id'] = run_id
    df['run_path'] = str(csv_path.parent)
    return df


def merge_grid_results(run_dirs: list) -> pd.DataFrame:
    """Merge grid_results.csv from multiple directories."""
    all_dfs = []
    
    for run_dir in run_dirs:
        run_path = Path(run_dir).resolve()
        csv_path = run_path / "grid_results.csv"
        
        if not csv_path.exists():
            print(f"[WARN] No grid_results.csv in {run_dir}", file=sys.stderr)
            continue
        
        run_id = extract_run_id(str(run_path))
        df = load_grid_results(csv_path, run_id)
        all_dfs.append(df)
        print(f"[OK] Loaded {len(df)} rows from {run_dir}")
    
    if not all_dfs:
        print("[ERROR] No grid_results.csv files found", file=sys.stderr)
        sys.exit(1)
    
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values(['discrete_lr', 'continuous_lr', 'run_id'])
    merged['n_trials'] = merged.groupby(['discrete_lr', 'continuous_lr']).cumcount() + 1
    
    return merged


def analyze_subsets(merged_df: pd.DataFrame, max_n: int = None):
    """Run ANOVA on different subset sizes."""
    print("\n" + "=" * 50)
    print("ANOVA Results by Subset Size")
    print("=" * 50)
    
    if max_n is None:
        max_n = merged_df['n_trials'].max()
    
    results = []
    for n in range(2, max_n + 1):
        subset = merged_df[merged_df['n_trials'] <= n]
        
        group_counts = subset.groupby(['discrete_lr', 'continuous_lr']).size()
        if group_counts.min() < 2:
            continue
            
        groups = [group['mean_reward'].values 
                  for _, group in subset.groupby(['discrete_lr', 'continuous_lr'])]
        
        if len(groups) >= 2 and len(groups[0]) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            sig = "*" if p_value < 0.05 else ""
            print(f"n={n:2d}: F={f_stat:8.4f}, p={p_value:.6f} {sig}")
            results.append({'n': n, 'F': f_stat, 'p_value': p_value, 'significant': p_value < 0.05})
    
    return pd.DataFrame(results) if results else None


def print_summary(merged_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 50)
    print("Summary Statistics")
    print("=" * 50)
    
    print(f"Total rows: {len(merged_df)}")
    print(f"Unique cells: {len(merged_df.groupby(['discrete_lr', 'continuous_lr']))}")
    print(f"Max trials per cell: {merged_df['n_trials'].max()}")
    print(f"Runs included: {list(merged_df['run_id'].unique())}")
    
    print("\n--- Per-Cell Statistics ---")
    cell_stats = merged_df.groupby(['discrete_lr', 'continuous_lr']).agg({
        'mean_reward': ['mean', 'std', 'count']
    }).round(4)
    print(cell_stats)


def create_visualizations(merged_df: pd.DataFrame, output_dir: str = "."):
    """Create heatmap and other visualizations."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    max_n = merged_df['n_trials'].max()
    latest_data = merged_df[merged_df['n_trials'] == max_n]
    
    pivot = latest_data.pivot(
        index='continuous_lr', 
        columns='discrete_lr', 
        values='mean_reward'
    )
    pivot = pivot.sort_index(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", 
                ax=axes[0], cbar_kws={'label': 'Mean Reward'})
    axes[0].set_title(f"Mean Reward (n={max_n} trials per cell)\n{output_dir}")
    axes[0].set_xlabel("Discrete Learning Rate")
    axes[0].set_ylabel("Continuous Learning Rate")
    
    std_data = latest_data.groupby(['continuous_lr', 'discrete_lr'])['mean_reward'].std().unstack()
    sns.heatmap(std_data, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=axes[1], cbar_kws={'label': 'Std Dev'})
    axes[1].set_title(f"Standard Deviation (n={max_n})\nLower is more consistent")
    axes[1].set_xlabel("Discrete Learning Rate")
    axes[1].set_ylabel("Continuous Learning Rate")
    
    plt.tight_layout()
    heatmap_path = f"{output_dir}/{timestamp}-heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Saved heatmap to {heatmap_path}")
    plt.close()
    
    anova_results = analyze_subsets(merged_df)
    if anova_results is not None and len(anova_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(anova_results['n'], anova_results['p_value'], 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=0.05, color='r', linestyle='--', label='p = 0.05 significance threshold')
        ax.fill_between(anova_results['n'], 0, 0.05, alpha=0.2, color='green', label='Significant region')
        ax.set_xlabel('Number of Trials per Cell (n)')
        ax.set_ylabel('ANOVA p-value')
        ax.set_title('ANOVA Significance vs Sample Size\nDetecting differences between hyperparameter combos')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        anova_path = f"{output_dir}/{timestamp}-anova-convergence.png"
        plt.savefig(anova_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved ANOVA convergence plot to {anova_path}")
        plt.close()
    
    print("[INFO] Visualizations complete!")


def plot_bo_progress(run_dir):
    """Plot BO progress from best_scores_history.csv and wip-client.csv.
    
    Shows:
    - Complete trials as circles (o)
    - In-progress trials as crosses (x)
    - Best so far as dashed line
    """
    run_path = Path(run_dir)
    
    # Load complete trial history
    history_path = run_path / "best_scores_history.csv"
    wip_path = run_path / "wip-client.csv"
    
    if not history_path.exists() and not wip_path.exists():
        print(f"[ERROR] No best_scores_history.csv or wip-client.csv found in {run_dir}")
        return
    
    # Get completed trials from history
    completed_trials = {}
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        for _, row in history_df.iterrows():
            completed_trials[int(row['trial'])] = {
                'mean_reward': row['mean_reward'],
                'best_so_far': row['best_so_far']
            }
    
    # Get all trials from wip-client.csv for status
    in_progress_trials = {}
    if wip_path.exists():
        wip_df = pd.read_csv(wip_path)
        for _, row in wip_df.iterrows():
            trial_idx = int(row['trial_index'])
            if trial_idx not in completed_trials:
                in_progress_trials[trial_idx] = {
                    'mean_reward': row['mean_reward'],
                    'status': row['trial_status']
                }
    
    # Combine and sort all trials
    all_trials = {}
    for trial_idx, data in completed_trials.items():
        all_trials[trial_idx] = {**data, 'complete': True}
    for trial_idx, data in in_progress_trials.items():
        if trial_idx not in all_trials:
            all_trials[trial_idx] = {**data, 'complete': False}
    
    if not all_trials:
        print("[WARN] No trial data found")
        return
    
    sorted_trials = sorted(all_trials.keys())
    
    # Prepare plot data
    complete_x, complete_y = [], []
    inprogress_x, inprogress_y = [], []
    best_so_far_x, best_so_far_y = [], []
    current_best = 0.0
    
    for trial_idx in sorted_trials:
        data = all_trials[trial_idx]
        if data['complete']:
            complete_x.append(trial_idx)
            complete_y.append(data['mean_reward'])
            if data['mean_reward'] > current_best:
                current_best = data['mean_reward']
        else:
            inprogress_x.append(trial_idx)
            inprogress_y.append(data['mean_reward'])
        best_so_far_x.append(trial_idx)
        best_so_far_y.append(current_best)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot best so far first (dashed line)
    ax.plot(best_so_far_x, best_so_far_y, 'g--', linewidth=2, label='Best So Far', zorder=1)
    
    # Plot complete trials as circles
    if complete_x:
        ax.scatter(complete_x, complete_y, c='blue', s=100, marker='o', 
                   label=f'Complete ({len(complete_x)})', zorder=3)
        ax.plot(complete_x, complete_y, 'b-', alpha=0.3, zorder=1)
    
    # Plot in-progress trials as crosses
    if inprogress_x:
        ax.scatter(inprogress_x, inprogress_y, c='red', s=150, marker='x', 
                   linewidths=2, label=f'In Progress ({len(inprogress_x)})', zorder=3)
    
    # Annotate points with values
    for trial_idx in sorted_trials:
        data = all_trials[trial_idx]
        label = f"{data['mean_reward']:.4f}"
        offset = 0.02 if data['complete'] else -0.03
        ax.annotate(label, (trial_idx, data['mean_reward'] + offset), 
                   fontsize=8, ha='center', va='bottom' if data['complete'] else 'top')
    
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Mean Reward')
    total = len(complete_x) + len(inprogress_x)
    ax.set_title(f'Bayesian Optimization Progress ({len(complete_x)}/{total} complete)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Save (overwrite best-scores-live.png)
    output_path = run_path / "best-scores-live.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved BO progress plot to {output_path}")
    plt.close()
    
    # Print summary
    print(f"\n--- BO Progress Summary ---")
    print(f"Total trials: {total}")
    print(f"Complete: {len(complete_x)}")
    print(f"In progress: {len(inprogress_x)}")
    if complete_y:
        print(f"Current best: {current_best:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Merge, analyze, and visualize grid search results")
    parser.add_argument("run_dirs", nargs="*", help="Run directories to merge (or glob pattern)")
    parser.add_argument("--bo-progress", metavar="DIR", help="Show BO progress plot (specify run directory)")
    parser.add_argument("--output", "-o", default="merged_grid_results.csv", help="Output filename")
    parser.add_argument("--no-analyze", action="store_true", help="Skip ANOVA analysis")
    parser.add_argument("--visualize", "-v", action="store_true", default=True, help="Create visualizations")
    parser.add_argument("--max-n", type=int, default=None, help="Maximum n to include")
    
    args = parser.parse_args()
    
    # BO progress mode (standalone)
    if args.bo_progress:
        plot_bo_progress(args.bo_progress)
        return
    
    # Grid search analysis mode
    if not args.run_dirs:
        print("[ERROR] Please specify run directories or use --bo-progress")
        parser.print_help()
        sys.exit(1)
    
    print(f"Merging {len(args.run_dirs)} run directories...")
    
    merged = merge_grid_results(args.run_dirs)
    print_summary(merged)
    
    if not args.no_analyze:
        analyze_subsets(merged, max_n=args.max_n)
    
    merged.to_csv(args.output, index=False)
    print(f"\n[INFO] Saved merged results to {args.output}")
    
    if args.visualize:
        output_dir = Path(args.output).parent if Path(args.output).parent != Path('.') else "."
        create_visualizations(merged, output_dir=str(output_dir))


if __name__ == "__main__":
    main()
