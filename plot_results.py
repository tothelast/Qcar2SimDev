import json
import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    # Load results
    try:
        with open("ade_results_all_epochs.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: ade_results_all_epochs.json not found. Run calculate_ade.py first.")
        return

    # Support both old flat format and new nested format
    if "policy_ade" in results:
        policy_results = results["policy_ade"]
        expert_ade = results.get("expert_ade")
    else:
        # Legacy flat format (epoch keys at top level)
        policy_results = results
        expert_ade = None

    # Extract policy ADE data
    epochs = []
    ades = []

    sorted_keys = sorted(policy_results.keys(),
                         key=lambda x: int(x.split('=')[-1]) if '=' in x else 999)

    for key in sorted_keys:
        if "epoch=" in key:
            epoch_num = int(key.split('=')[-1])
            epochs.append(epoch_num)
            ades.append(policy_results[key])

    # Plot
    plt.figure(figsize=(10, 6))

    # Ground truth route baseline at y=0
    plt.axhline(y=0, color='g', linestyle='--', linewidth=2,
                label='Ground Truth Route (ADE=0)')

    # Expert demonstration ADE (constant horizontal line)
    if expert_ade is not None:
        plt.axhline(y=expert_ade, color='r', linestyle='--', linewidth=1.5,
                    label=f'Expert Demonstration (ADE={expert_ade:.3f})')

    # Learned policy ADE curve
    plt.plot(epochs, ades, marker='o', linestyle='-', color='b',
             label='Learned Policy ADE')

    plt.xlabel('Training Epochs')
    plt.ylabel('Average Displacement Error (ADE)')
    plt.title('Policy & Expert Performance vs Ground Truth')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig('policy_vs_expert_curve.png', dpi=150, bbox_inches='tight')
    print("Plot saved to policy_vs_expert_curve.png")

if __name__ == "__main__":
    plot_results()
