"""
Populate LaTeX table value files for Chapter 7 from test results.

This script reads the SimLingo and ACC test results JSON files and generates
small .tex files in report/figures/ that are \input{} by the results chapter.

Usage:
    python results/populate_latex_tables.py \
        --simlingo-results results/test_results_YYYYMMDD_HHMMSS.json \
        --acc-results results/acc_test_results_YYYYMMDD_HHMMSS.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results(path):
    with open(path) as f:
        return json.load(f)


def group_by_scenario(results):
    grouped = defaultdict(list)
    for r in results['results']:
        grouped[r['scenario_name']].append(r)
    return grouped


def write_val(path, value):
    """Write a single LaTeX value to a file (no newline, no trailing space)."""
    with open(path, 'w') as f:
        f.write(str(value))


def main():
    parser = argparse.ArgumentParser(description='Populate LaTeX table values')
    parser.add_argument('--simlingo-results', type=str, required=True)
    parser.add_argument('--acc-results', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='report/figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = load_results(args.simlingo_results)
    acc = load_results(args.acc_results)

    sim_grouped = group_by_scenario(sim)
    acc_grouped = group_by_scenario(acc)

    # --- Baseline table values ---
    for prefix, grouped in [('sim', sim_grouped), ('acc', acc_grouped)]:
        baseline = grouped.get('baseline', [])
        if baseline:
            coverages = [r['route_coverage_percent'] for r in baseline]
            avg_devs = [r['avg_lateral_deviation'] for r in baseline]
            max_devs = [r['max_lateral_deviation'] for r in baseline]
            times = [r['total_time'] for r in baseline]

            write_val(output_dir / f'val_{prefix}_baseline_coverage',
                      f'{np.mean(coverages):.1f}')
            write_val(output_dir / f'val_{prefix}_baseline_avgdev',
                      f'{np.mean(avg_devs):.3f}')
            write_val(output_dir / f'val_{prefix}_baseline_maxdev',
                      f'{np.mean(max_devs):.3f}')
            write_val(output_dir / f'val_{prefix}_baseline_time',
                      f'{np.mean(times):.1f}')

    # --- Obstacle results table rows ---
    obstacle_scenarios = ['obstacle_var1', 'obstacle_var2', 'obstacle_var3',
                          'obstacle_var4', 'obstacle_var5']
    var_labels = {
        'obstacle_var1': 'Var 1 (Early)',
        'obstacle_var2': 'Var 2 (Mid)',
        'obstacle_var3': 'Var 3 (Exit)',
        'obstacle_var4': 'Var 4 (Straight)',
        'obstacle_var5': 'Var 5 (Late)',
    }

    rows = []
    for scenario in obstacle_scenarios:
        label = var_labels[scenario]
        for ctrl_name, grouped in [('SimLingo', sim_grouped), ('ACC', acc_grouped)]:
            results = grouped.get(scenario, [])
            if results:
                coverage = np.mean([r['route_coverage_percent'] for r in results])
                collisions = sum(1 for r in results if r['safety']['collision_detected'])
                stopped = sum(1 for r in results if r['safety']['stopped_before_obstacle'])
                stop_dists = [r['safety']['stopping_distance'] for r in results
                              if r['safety']['stopped_before_obstacle'] and r['safety']['stopping_distance'] > 0]
                avg_stop_dist = np.mean(stop_dists) if stop_dists else -1
                avg_lat = np.mean([r['avg_lateral_deviation'] for r in results])
                n = len(results)

                collision_str = f'{collisions}/{n}'
                stopped_str = f'{stopped}/{n}'
                stop_dist_str = f'{avg_stop_dist:.1f}' if avg_stop_dist > 0 else 'N/A'

                row = f'{label} & {ctrl_name} & {coverage:.1f} & {collision_str} & {stopped_str} & {stop_dist_str} & {avg_lat:.3f} \\\\'
                rows.append(row)
            else:
                row = f'{label} & {ctrl_name} & N/A & N/A & N/A & N/A & N/A \\\\'
                rows.append(row)

        # Add midrule between variants (but not after last)
        if scenario != obstacle_scenarios[-1]:
            rows.append('\\midrule')

    # Include \bottomrule in the generated file to avoid \noalign errors
    # when \input is used inside \resizebox tabular environments.
    rows.append('\\bottomrule')

    with open(output_dir / 'obstacle_results_table_rows', 'w') as f:
        f.write('\n'.join(rows))

    print(f"LaTeX table values written to {output_dir}")
    print(f"  - Baseline values: val_sim_baseline_*, val_acc_baseline_*")
    print(f"  - Obstacle table: obstacle_results_table_rows")


if __name__ == '__main__':
    main()
