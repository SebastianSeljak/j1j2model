import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_results(results, exact_energy, n_spins):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.axhline(y=exact_energy, color='black', linestyle='--', linewidth=2, label=f'Exact ({exact_energy:.4f})')
    
    colors = ['#1f77b4', '#ff7f0e']
    for i, (name, data) in enumerate(results.items()):
        means = np.array(data['mean'])
        stds = np.array(data['std'])
        epochs = range(1, len(means) + 1)
        plt.plot(epochs, means, label=name, color=colors[i], linewidth=2)
        plt.fill_between(epochs, means - stds, means + stds, color=colors[i], alpha=0.15)

    plt.xlabel("Epoch")
    plt.ylabel("Energy <H>")
    plt.title(f"VMC Optimization Comparison (N={n_spins})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("vmc_comparison_robust.png")
    print("\nPlot saved to vmc_comparison_robust.png")
    plt.show()


def plot_j2_comparison(results_data, n_spins):
    """
    Creates a 1x3 subplot figure comparing J2=0, 0.5, 1.0
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    
    j2_values = sorted(results_data.keys())
    
    for i, J2 in enumerate(j2_values):
        ax = axes[i]
        data = results_data[J2]
        
        # Plot Exact
        ax.axhline(y=data['exact'], color='k', linestyle='--', linewidth=2, label=f"Exact ({data['exact']:.2f})")
        
        # Plot Models
        ax.plot(data['cnn'], label="Real CNN (Sign Rule)", color="#1f77b4", linewidth=2, alpha=0.8)
        ax.plot(data['complexcnn'], label="Complex Sine (Learned)", color="#d62728", linewidth=2, alpha=0.8)
        
        ax.set_title(f"J2 = {J2}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epochs")
        if i == 0: ax.set_ylabel("Energy <H>")
        ax.legend(loc='upper right')
        
    plt.suptitle(f"VMC Model Comparison (N={n_spins} Particles)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig("j2_comparison_12spins.png", dpi=150, bbox_inches='tight')
    plt.show()