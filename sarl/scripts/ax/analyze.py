#!/usr/bin/env python3
"""
Analyze and visualize grid search results.

Usage:
    python analyze.py runs/2026-03-16_03-00
    python analyze.py runs/2026-03-16_03-00 runs/2026-03-16_05-00
    python analyze.py "runs/*/"
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


def main():
    parser = argparse.ArgumentParser(description="Merge, analyze, and visualize grid search results")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to merge (or glob pattern)")
    parser.add_argument("--output", "-o", default="merged_grid_results.csv", help="Output filename")
    parser.add_argument("--no-analyze", action="store_true", help="Skip ANOVA analysis")
    parser.add_argument("--visualize", "-v", action="store_true", default=True, help="Create visualizations")
    parser.add_argument("--max-n", type=int, default=None, help="Maximum n to include")
    
    args = parser.parse_args()
    
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
