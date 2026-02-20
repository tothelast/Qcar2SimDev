"""
Generate report figures from SimLingo and ACC baseline test results.

Usage:
    python results/generate_report_figures.py \
        --simlingo-results results/test_results_YYYYMMDD_HHMMSS.json \
        --acc-results results/acc_test_results_YYYYMMDD_HHMMSS.json \
        --output-dir report/figures
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

SIMLINGO_COLOR = '#2196F3'  # Blue
ACC_COLOR = '#FF9800'       # Orange
PASS_COLOR = '#4CAF50'      # Green
FAIL_COLOR = '#F44336'      # Red


def load_results(path):
    """Load test results JSON file."""
    with open(path) as f:
        return json.load(f)


def group_by_scenario(results):
    """Group results by scenario name."""
    grouped = defaultdict(list)
    for r in results['results']:
        grouped[r['scenario_name']].append(r)
    return grouped


def fig1_route_coverage_comparison(simlingo, acc, output_dir):
    """Bar chart comparing route coverage across scenarios."""
    sim_grouped = group_by_scenario(simlingo)
    acc_grouped = group_by_scenario(acc)

    scenarios = ['baseline', 'obstacle_var1', 'obstacle_var2',
                 'obstacle_var3', 'obstacle_var4', 'obstacle_var5']
    labels = ['Baseline', 'Var 1\n(Early)', 'Var 2\n(Mid)', 'Var 3\n(Exit)',
              'Var 4\n(Straight)', 'Var 5\n(Late)']

    sim_means, sim_stds = [], []
    acc_means, acc_stds = [], []

    for s in scenarios:
        sim_vals = [r['route_coverage_percent'] for r in sim_grouped.get(s, [])]
        acc_vals = [r['route_coverage_percent'] for r in acc_grouped.get(s, [])]
        sim_means.append(np.mean(sim_vals) if sim_vals else 0)
        sim_stds.append(np.std(sim_vals) if len(sim_vals) > 1 else 0)
        acc_means.append(np.mean(acc_vals) if acc_vals else 0)
        acc_stds.append(np.std(acc_vals) if len(acc_vals) > 1 else 0)

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, sim_means, width, yerr=sim_stds, capsize=4,
                   label='SimLingo', color=SIMLINGO_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width/2, acc_means, width, yerr=acc_stds, capsize=4,
                   label='ACC Baseline', color=ACC_COLOR, alpha=0.85)

    ax.set_ylabel('Route Coverage (%)')
    ax.set_title('Route Coverage: SimLingo vs ACC Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'route_coverage_comparison.png')
    plt.close()
    print(f"Saved: route_coverage_comparison.png")


def fig2_safety_comparison(simlingo, acc, output_dir):
    """Grouped bar chart of safety metrics for obstacle scenarios."""
    sim_grouped = group_by_scenario(simlingo)
    acc_grouped = group_by_scenario(acc)

    obstacle_scenarios = ['obstacle_var1', 'obstacle_var2', 'obstacle_var3',
                          'obstacle_var4', 'obstacle_var5']

    def get_safety_stats(grouped):
        obs_results = []
        for s in obstacle_scenarios:
            obs_results.extend(grouped.get(s, []))
        if not obs_results:
            return 0, 0, 0
        n = len(obs_results)
        collision_rate = sum(1 for r in obs_results if r['safety']['collision_detected']) / n * 100
        stop_rate = sum(1 for r in obs_results if r['safety']['stopped_before_obstacle']) / n * 100
        stop_dists = [r['safety']['stopping_distance'] for r in obs_results
                      if r['safety']['stopped_before_obstacle'] and r['safety']['stopping_distance'] > 0]
        avg_stop_dist = np.mean(stop_dists) if stop_dists else 0
        return collision_rate, stop_rate, avg_stop_dist

    sim_collision, sim_stop, sim_dist = get_safety_stats(sim_grouped)
    acc_collision, acc_stop, acc_dist = get_safety_stats(acc_grouped)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    # Collision rate
    ax = axes[0]
    bars = ax.bar(['SimLingo', 'ACC'], [sim_collision, acc_collision],
                  color=[SIMLINGO_COLOR, ACC_COLOR], alpha=0.85)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Collision Rate')
    ax.set_ylim(0, max(110, max(sim_collision, acc_collision) * 1.2))
    for bar, val in zip(bars, [sim_collision, acc_collision]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=11)

    # Stop success rate
    ax = axes[1]
    bars = ax.bar(['SimLingo', 'ACC'], [sim_stop, acc_stop],
                  color=[SIMLINGO_COLOR, ACC_COLOR], alpha=0.85)
    ax.set_ylabel('Rate (%)')
    ax.set_title('Stop Success Rate')
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, [sim_stop, acc_stop]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=11)

    # Avg stopping distance
    ax = axes[2]
    bars = ax.bar(['SimLingo', 'ACC'], [sim_dist, acc_dist],
                  color=[SIMLINGO_COLOR, ACC_COLOR], alpha=0.85)
    ax.set_ylabel('Distance (m)')
    ax.set_title('Avg Stopping Distance')
    ax.set_ylim(0, max(sim_dist, acc_dist) * 1.25)
    for bar, val in zip(bars, [sim_dist, acc_dist]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}m', ha='center', va='bottom', fontsize=11)

    plt.suptitle('Safety Metrics: Obstacle Scenarios', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'safety_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: safety_comparison.png")


def fig3_lateral_deviation(simlingo, acc, output_dir):
    """Bar chart comparing lateral deviation across scenarios."""
    sim_grouped = group_by_scenario(simlingo)
    acc_grouped = group_by_scenario(acc)

    scenarios = ['baseline', 'obstacle_var1', 'obstacle_var2',
                 'obstacle_var3', 'obstacle_var4', 'obstacle_var5']
    labels = ['Baseline', 'Var 1', 'Var 2', 'Var 3', 'Var 4', 'Var 5']

    sim_means, acc_means = [], []
    for s in scenarios:
        sim_vals = [r['avg_lateral_deviation'] for r in sim_grouped.get(s, [])]
        acc_vals = [r['avg_lateral_deviation'] for r in acc_grouped.get(s, [])]
        sim_means.append(np.mean(sim_vals) if sim_vals else 0)
        acc_means.append(np.mean(acc_vals) if acc_vals else 0)

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, sim_means, width, label='SimLingo', color=SIMLINGO_COLOR, alpha=0.85)
    ax.bar(x + width/2, acc_means, width, label='ACC Baseline', color=ACC_COLOR, alpha=0.85)

    ax.set_ylabel('Avg Lateral Deviation (m)')
    ax.set_title('Lateral Deviation: SimLingo vs ACC Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'lateral_deviation_comparison.png')
    plt.close()
    print(f"Saved: lateral_deviation_comparison.png")


def fig4_trajectory_overlay(simlingo, acc, output_dir):
    """Bird's-eye trajectory overlays for each obstacle variant."""
    # Load route waypoints
    route_path = Path(__file__).parent.parent / 'config/routes/roundabout_navigation.json'
    with open(route_path) as f:
        route_data = json.load(f)
    route_wps = np.array(route_data['waypoints'])

    obstacle_locations = {
        'obstacle_var1': [21.01, 33.90],
        'obstacle_var2': [18.85, 44.23],
        'obstacle_var3': [6.07, 44.97],
        'obstacle_var4': [-10.60, 44.97],
        'obstacle_var5': [-18.73, 40.37],
    }

    sim_grouped = group_by_scenario(simlingo)
    acc_grouped = group_by_scenario(acc)

    obstacle_scenarios = ['obstacle_var1', 'obstacle_var2', 'obstacle_var3',
                          'obstacle_var4', 'obstacle_var5']
    var_labels = ['Var 1 (Early)', 'Var 2 (Mid)', 'Var 3 (Exit)',
                  'Var 4 (Straight)', 'Var 5 (Late)']

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for idx, (scenario, label) in enumerate(zip(obstacle_scenarios, var_labels)):
        ax = axes[idx]

        # Route centerline
        ax.plot(route_wps[:, 0], route_wps[:, 1], 'k--', linewidth=1, alpha=0.4, label='Route')

        # Obstacle
        obs_key = scenario
        if obs_key in obstacle_locations:
            obs = obstacle_locations[obs_key]
            ax.plot(obs[0], obs[1], 'rs', markersize=10, markeredgewidth=2, label='Obstacle')

        # SimLingo trajectories
        for i, r in enumerate(sim_grouped.get(scenario, [])):
            traj_path = Path(r['trajectory_log_path'])
            if traj_path.exists():
                with open(traj_path) as f:
                    traj_data = json.load(f)
                positions = np.array([t['position'][:2] for t in traj_data['trajectory']])
                if len(positions) > 0:
                    lbl = 'SimLingo' if i == 0 else None
                    ax.plot(positions[:, 0], positions[:, 1], '-',
                            color=SIMLINGO_COLOR, linewidth=1.5, alpha=0.7, label=lbl)

        # ACC trajectories
        for i, r in enumerate(acc_grouped.get(scenario, [])):
            traj_path = Path(r['trajectory_log_path'])
            if traj_path.exists():
                with open(traj_path) as f:
                    traj_data = json.load(f)
                positions = np.array([t['position'][:2] for t in traj_data['trajectory']])
                if len(positions) > 0:
                    lbl = 'ACC' if i == 0 else None
                    ax.plot(positions[:, 0], positions[:, 1], '--',
                            color=ACC_COLOR, linewidth=1.5, alpha=0.7, label=lbl)

        ax.set_aspect('equal')
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc='best')

    plt.suptitle('Trajectory Overlays by Obstacle Variant', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_overlays.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: trajectory_overlays.png")


def fig5_pass_fail_summary(simlingo, acc, output_dir):
    """Pass/fail summary table as a figure."""
    scenarios = ['baseline', 'obstacle_var1', 'obstacle_var2',
                 'obstacle_var3', 'obstacle_var4', 'obstacle_var5']
    labels = ['Baseline', 'Obstacle Var 1', 'Obstacle Var 2',
              'Obstacle Var 3', 'Obstacle Var 4', 'Obstacle Var 5']

    sim_grouped = group_by_scenario(simlingo)
    acc_grouped = group_by_scenario(acc)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Build table data
    header = ['Scenario', 'SimLingo\nPass Rate', 'ACC Baseline\nPass Rate']
    table_data = []

    for s, label in zip(scenarios, labels):
        sim_results = sim_grouped.get(s, [])
        acc_results = acc_grouped.get(s, [])
        sim_pass = sum(1 for r in sim_results if r['pass_status'])
        acc_pass = sum(1 for r in acc_results if r['pass_status'])
        sim_total = len(sim_results)
        acc_total = len(acc_results)
        sim_str = f"{sim_pass}/{sim_total}" if sim_total > 0 else "N/A"
        acc_str = f"{acc_pass}/{acc_total}" if acc_total > 0 else "N/A"
        table_data.append([label, sim_str, acc_str])

    # Aggregate
    sim_all = simlingo['results']
    acc_all = acc['results']
    sim_total_pass = sum(1 for r in sim_all if r['pass_status'])
    acc_total_pass = sum(1 for r in acc_all if r['pass_status'])
    table_data.append(['TOTAL',
                       f"{sim_total_pass}/{len(sim_all)} ({sim_total_pass/len(sim_all)*100:.0f}%)",
                       f"{acc_total_pass}/{len(acc_all)} ({acc_total_pass/len(acc_all)*100:.0f}%)"])

    table = ax.table(cellText=table_data, colLabels=header,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Give the header row extra height for the two-line text
    for j in range(3):
        table[0, j].set_height(table[0, j].get_height() * 1.4)

    # Color coding
    for i, row in enumerate(table_data):
        for j in range(1, 3):
            cell = table[i + 1, j]
            val_str = row[j]
            if '/' in val_str:
                parts = val_str.split('/')
                if parts[0] == parts[1].split()[0]:
                    cell.set_facecolor('#C8E6C9')  # Light green
                elif parts[0] == '0':
                    cell.set_facecolor('#FFCDD2')  # Light red
                else:
                    cell.set_facecolor('#FFF9C4')  # Light yellow

    # Header styling
    for j in range(3):
        table[0, j].set_facecolor('#E3F2FD')
        table[0, j].set_text_props(fontweight='bold')

    # Last row (total) styling
    for j in range(3):
        table[len(table_data), j].set_text_props(fontweight='bold')

    ax.set_title('Pass/Fail Summary: SimLingo vs ACC Baseline', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_fail_summary.png', bbox_inches='tight')
    plt.close()
    print(f"Saved: pass_fail_summary.png")


def main():
    parser = argparse.ArgumentParser(description='Generate report figures from test results')
    parser.add_argument('--simlingo-results', type=str, required=True,
                        help='Path to SimLingo test results JSON')
    parser.add_argument('--acc-results', type=str, required=True,
                        help='Path to ACC baseline test results JSON')
    parser.add_argument('--output-dir', type=str, default='report/figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    simlingo = load_results(args.simlingo_results)
    acc = load_results(args.acc_results)

    print(f"SimLingo: {len(simlingo['results'])} runs, pass rate: {simlingo['pass_rate']*100:.0f}%")
    print(f"ACC:      {len(acc['results'])} runs, pass rate: {acc['pass_rate']*100:.0f}%")

    print("\nGenerating figures...")
    fig1_route_coverage_comparison(simlingo, acc, output_dir)
    fig2_safety_comparison(simlingo, acc, output_dir)
    fig3_lateral_deviation(simlingo, acc, output_dir)
    fig4_trajectory_overlay(simlingo, acc, output_dir)
    fig5_pass_fail_summary(simlingo, acc, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
