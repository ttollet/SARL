#!/usr/bin/env python3
"""
Statistical t-test analysis for grid search results.

Performs both independent and paired t-tests on grid search trial results.

Usage:
    python _ttest_grid.py                                      # Interactive mode
    python _ttest_grid.py runs/grid/proper/complete/2026-03-17_03-50/seed_results.csv
    python _ttest_grid.py --trial1 0 --trial2 1               # By trial index
    python _ttest_grid.py --all-pairs                         # Compare all pairs

References:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_interpretation(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def run_ttest(df, trial1_idx, trial2_idx, alpha=0.05):
    """Run both independent and paired t-tests between two trials."""
    rewards1 = df[df["trial_index"] == trial1_idx]["mean_reward"].values
    rewards2 = df[df["trial_index"] == trial2_idx]["mean_reward"].values

    params1 = df[df["trial_index"] == trial1_idx].iloc[0]
    params2 = df[df["trial_index"] == trial2_idx].iloc[0]

    print(f"\n{'=' * 60}")
    print(f"Comparing trial {trial1_idx} vs trial {trial2_idx}")
    print(f"{'=' * 60}")
    print(
        f"Trial {trial1_idx}: d_lr={params1['discrete_lr']:.0e}, c_lr={params1['continuous_lr']:.0e}, u={params1['update_ratio']}"
    )
    print(
        f"  n={len(rewards1)}, mean={np.mean(rewards1):.4f}, std={np.std(rewards1, ddof=1):.4f}"
    )
    print(
        f"Trial {trial2_idx}: d_lr={params2['discrete_lr']:.0e}, c_lr={params2['continuous_lr']:.0e}, u={params2['update_ratio']}"
    )
    print(
        f"  n={len(rewards2)}, mean={np.mean(rewards2):.4f}, std={np.std(rewards2, ddof=1):.4f}"
    )

    mean_diff = np.mean(rewards1) - np.mean(rewards2)
    print(f"\nMean difference: {mean_diff:.4f}")

    t_ind, p_ind = stats.ttest_ind(rewards1, rewards2)
    d_ind = cohens_d(rewards1, rewards2)
    print(f"\n--- Independent samples t-test ---")
    print(f"t-statistic: {t_ind:.4f}")
    print(f"p-value: {p_ind:.6f}")
    print(f"Cohen's d: {d_ind:.4f} ({effect_size_interpretation(d_ind)})")
    print(f"Significant at α={alpha}: {'YES' if p_ind < alpha else 'NO'}")

    if len(rewards1) == len(rewards2):
        t_paired, p_paired = stats.ttest_rel(rewards1, rewards2)
        d_paired = cohens_d(rewards1, rewards2)
        print(f"\n--- Paired samples t-test ---")
        print(f"t-statistic: {t_paired:.4f}")
        print(f"p-value: {p_paired:.6f}")
        print(f"Cohen's d: {d_paired:.4f} ({effect_size_interpretation(d_paired)})")
        print(f"Significant at α={alpha}: {'YES' if p_paired < alpha else 'NO'}")

        ci = stats.t.interval(
            1 - alpha,
            len(rewards1) - 1,
            loc=mean_diff,
            scale=stats.sem(rewards1 - rewards2),
        )
        print(f"95% CI of difference: [{ci[0]:.4f}, {ci[1]:.4f}]")
    else:
        print("\n--- Paired t-test skipped: different sample sizes ---")

    return {
        "trial1": trial1_idx,
        "trial2": trial2_idx,
        "mean1": np.mean(rewards1),
        "mean2": np.mean(rewards2),
        "mean_diff": mean_diff,
        "t_independent": t_ind,
        "p_independent": p_ind,
        "cohens_d": d_ind,
        "significant": p_ind < alpha,
    }


def run_all_pairwise(df, alpha=0.05):
    """Run t-tests on all pairs of trials."""
    trial_indices = sorted(df["trial_index"].unique())
    results = []

    print(f"\nRunning pairwise t-tests on {len(trial_indices)} trials...")
    print(f"Total pairs: {len(trial_indices) * (len(trial_indices) - 1) // 2}")

    for i, t1 in enumerate(trial_indices):
        for t2 in trial_indices[i + 1 :]:
            result = run_ttest(df, t1, t2, alpha)
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_independent")
    return results_df


def list_available_runs():
    """List available grid search result directories."""
    runs_dir = Path("runs/grid")
    if not runs_dir.exists():
        print("No runs/grid directory found")
        return []

    runs = []
    for complete_dir in runs_dir.glob("*/complete/*"):
        seed_results = complete_dir / "seed_results.csv"
        if seed_results.exists():
            runs.append(seed_results)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="T-test analysis for grid search results"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Path to seed_results.csv (or auto-detect if not provided)",
    )
    parser.add_argument(
        "--trial1",
        "--t1",
        type=int,
        default=None,
        help="First trial index to compare",
    )
    parser.add_argument(
        "--trial2",
        "--t2",
        type=int,
        default=None,
        help="Second trial index to compare",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Run t-tests on all pairs of trials",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for results",
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        runs = list_available_runs()
        if not runs:
            print("No seed_results.csv found. Run grid search first.")
            return
        csv_path = runs[-1]
        print(f"Using latest run: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Trials: {sorted(df['trial_index'].unique())}")

    if args.all_pairs:
        results_df = run_all_pairwise(df, args.alpha)
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"Saved results to {args.output}")
        else:
            print("\n--- Summary (sorted by p-value) ---")
            print(results_df.to_string(index=False))
    elif args.trial1 is not None and args.trial2 is not None:
        run_ttest(df, args.trial1, args.trial2, args.alpha)
    else:
        print("\nInteractive mode - showing all trial statistics:")
        print("-" * 50)
        for trial_idx in sorted(df["trial_index"].unique()):
            trial_data = df[df["trial_index"] == trial_idx]
            params = trial_data.iloc[0]
            print(
                f"Trial {trial_idx}: d_lr={params['discrete_lr']:.0e}, "
                f"c_lr={params['continuous_lr']:.0e}, u={params['update_ratio']} "
                f"-> mean={trial_data['mean_reward'].mean():.4f} ± {trial_data['mean_reward'].std(ddof=1):.4f}"
            )
        print(
            "\nSpecify --trial1 and --trial2 to compare, or --all-pairs for all comparisons"
        )


if __name__ == "__main__":
    main()
