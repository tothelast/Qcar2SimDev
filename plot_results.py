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

    # Extract data
    epochs = []
    ades = []
    
    # Sort by epoch number
    sorted_keys = sorted(results.keys(), key=lambda x: int(x.split('=')[-1]) if '=' in x else 999)
    
    for key in sorted_keys:
        if "epoch=" in key:
            epoch_num = int(key.split('=')[-1])
            epochs.append(epoch_num)
            ades.append(results[key])
            
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ades, marker='o', linestyle='-', color='b', label='Learned Policy ADE')
    plt.axhline(y=0, color='g', linestyle='--', label='Expert Baseline (ADE=0)')
    
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Displacement Error (ADE)')
    plt.title('Policy Performance vs Expert')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('policy_vs_expert_curve.png')
    print("Plot saved to policy_vs_expert_curve.png")

if __name__ == "__main__":
    plot_results()
