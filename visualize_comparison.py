"""
Visualization Tool: Compare Training Configurations
Shows expected behavior: Original vs Improved vs Ideal
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_training_comparison():
    """Generate comparison plots of different training configurations"""
    
    epochs = np.arange(1, 201)
    
    # Simulate different training curves
    # Original: oscillating
    original_dice = np.concatenate([
        np.linspace(0.1, 0.76, 40),  # Epochs 1-40: Good progress
        np.array([0.76, 0.45, 0.65, 0.52, 0.71, 0.42, 0.68, 0.55, 0.72, 0.48]) + np.random.randn(10)*0.05,  # 41-50: Oscillation
        np.linspace(0.6, 0.78, 50),  # 51-100: Noisy progress
        np.linspace(0.78, 0.80, 100),  # 101-200: Plateau
    ])[:200]
    
    # Improved: smooth
    improved_dice = np.concatenate([
        np.linspace(0.1, 0.72, 40),   # Epochs 1-40
        np.linspace(0.72, 0.85, 40),  # Epochs 41-80
        np.linspace(0.85, 0.90, 60),  # Epochs 81-140
        np.linspace(0.90, 0.92, 60),  # Epochs 141-200
    ])
    
    # Ideal: perfectly smooth
    ideal_dice = np.concatenate([
        np.linspace(0.1, 0.70, 40),
        np.linspace(0.70, 0.85, 40),
        np.linspace(0.85, 0.90, 60),
        np.linspace(0.90, 0.91, 60),
    ])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== SUBPLOT 1: Main Comparison =====
    ax = axes[0, 0]
    ax.plot(epochs[:len(original_dice)], original_dice, 'r-', linewidth=2.5, label='Original (Oscillating)', alpha=0.7)
    ax.plot(epochs, improved_dice, 'g-', linewidth=2.5, label='Improved (Fixed)', alpha=0.7)
    ax.plot(epochs, ideal_dice, 'b--', linewidth=2, label='Ideal', alpha=0.6)
    ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='Target (0.90)', alpha=0.7)
    ax.fill_between(epochs, 0, 1, alpha=0.1, color='gray')
    ax.set_title('Training Curves: Original vs Improved', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Dice', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 1])
    
    # Highlight oscillation region
    ax.axvspan(40, 60, alpha=0.1, color='red', label='Oscillation Period')
    
    # ===== SUBPLOT 2: Metric Smoothness (Epoch-to-Epoch Changes) =====
    ax = axes[0, 1]
    orig_diffs = np.abs(np.diff(original_dice))
    imp_diffs = np.abs(np.diff(improved_dice))
    ideal_diffs = np.abs(np.diff(ideal_dice))
    
    ax.plot(epochs[:-1], orig_diffs, 'r-', linewidth=2, label='Original', alpha=0.7)
    ax.plot(epochs[:-1], imp_diffs, 'g-', linewidth=2, label='Improved', alpha=0.7)
    ax.plot(epochs[:-1], ideal_diffs, 'b--', linewidth=2, label='Ideal', alpha=0.6)
    ax.axhline(y=0.02, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Good Threshold')
    ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='OK Threshold')
    ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bad Threshold')
    
    ax.set_title('Training Smoothness (Epoch-to-Epoch Changes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('|Δ Dice|', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 200])
    ax.set_yscale('log')
    
    # ===== SUBPLOT 3: Convergence Speed =====
    ax = axes[1, 0]
    
    # Calculate epochs to reach milestones
    milestones = [0.7, 0.8, 0.85, 0.9]
    orig_epochs_to_milestone = []
    imp_epochs_to_milestone = []
    
    for milestone in milestones:
        orig_idx = np.where(original_dice >= milestone)[0]
        imp_idx = np.where(improved_dice >= milestone)[0]
        
        orig_epochs_to_milestone.append(orig_idx[0] if len(orig_idx) > 0 else 200)
        imp_epochs_to_milestone.append(imp_idx[0] if len(imp_idx) > 0 else 200)
    
    x_pos = np.arange(len(milestones))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, orig_epochs_to_milestone, width, label='Original', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, imp_epochs_to_milestone, width, label='Improved', color='green', alpha=0.7)
    
    ax.set_title('Convergence Speed (Epochs to Reach Milestone)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dice Target', fontsize=12)
    ax.set_ylabel('Epochs Required', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{m:.2f}' for m in milestones])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # ===== SUBPLOT 4: Training Stability Heatmap =====
    ax = axes[1, 1]
    
    # Create bins showing stability in each epoch range
    epoch_bins = [(1, 50), (51, 100), (101, 150), (151, 200)]
    bin_labels = ['1-50', '51-100', '101-150', '151-200']
    
    orig_stabilities = []
    imp_stabilities = []
    
    for start, end in epoch_bins:
        orig_stability = np.std(original_dice[start-1:end])
        imp_stability = np.std(improved_dice[start-1:end])
        orig_stabilities.append(orig_stability)
        imp_stabilities.append(imp_stability)
    
    x_pos = np.arange(len(bin_labels))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, orig_stabilities, width, label='Original (Higher=Unstable)', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, imp_stabilities, width, label='Improved (Lower=Stable)', color='green', alpha=0.7)
    
    ax.set_title('Training Stability by Epoch Range (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch Range', fontsize=12)
    ax.set_ylabel('Standard Deviation of Dice', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Training Configuration Comparison: Original vs Improved', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('configuration_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comparison plot saved to: configuration_comparison.png")
    plt.show()

def plot_hyperparameter_effects():
    """Show effect of each hyperparameter change"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    epochs = np.arange(1, 151)
    
    # Effect 1: Batch Size
    ax = axes[0, 0]
    ax.plot(epochs, np.linspace(0.1, 0.75, 150) + np.random.randn(150)*0.08, 'r-', linewidth=2, label='Batch=4 (Noisy)')
    ax.plot(epochs, np.linspace(0.1, 0.75, 150) + np.random.randn(150)*0.02, 'g-', linewidth=2, label='Batch=8 (Smooth)')
    ax.set_title('Effect: Batch Size 4 → 8', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effect 2: Learning Rate
    ax = axes[0, 1]
    ax.plot(epochs, np.linspace(0.1, 0.65, 150), 'r-', linewidth=2, label='LR=5e-4 (Slow)')
    ax.plot(epochs, np.linspace(0.1, 0.75, 150), 'g-', linewidth=2, label='LR=1e-3 (Better)')
    ax.set_title('Effect: Learning Rate 5e-4 → 1e-3', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effect 3: Gradient Clipping
    ax = axes[0, 2]
    no_clip = np.linspace(0.1, 0.70, 150)
    no_clip[40:60] += np.random.randn(20) * 0.15  # Spikes
    ax.plot(epochs, no_clip, 'r-', linewidth=2, label='No Clipping (Spikes)')
    ax.plot(epochs, np.linspace(0.1, 0.75, 150), 'g-', linewidth=2, label='With Clipping (Smooth)')
    ax.set_title('Effect: Gradient Clipping', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effect 4: Accumulation Steps
    ax = axes[1, 0]
    ax.plot(epochs, np.linspace(0.1, 0.68, 150) + np.random.randn(150)*0.05, 'r-', linewidth=2, label='Accum=8 (Jerky LR)')
    ax.plot(epochs, np.linspace(0.1, 0.75, 150) + np.random.randn(150)*0.02, 'g-', linewidth=2, label='Accum=4 (Smooth LR)')
    ax.set_title('Effect: Accumulation Steps 8 → 4', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effect 5: Scheduler Warmup
    ax = axes[1, 1]
    lr_long_warmup = np.concatenate([np.linspace(1e-7, 1e-3, 45), np.linspace(1e-3, 1e-6, 105)])
    lr_short_warmup = np.concatenate([np.linspace(1e-7, 1e-3, 20), np.linspace(1e-3, 1e-6, 130)])
    ax.semilogy(epochs, lr_long_warmup, 'r-', linewidth=2, label='pct_start=0.30 (Long warmup)')
    ax.semilogy(epochs, lr_short_warmup, 'g-', linewidth=2, label='pct_start=0.15 (Short warmup)')
    ax.set_title('Effect: Scheduler Warmup 30% → 15%', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate (log)')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effect 6: Epochs
    ax = axes[1, 2]
    epochs_150 = np.linspace(0.1, 0.88, 150)
    epochs_200 = np.linspace(0.1, 0.92, 200)
    ax.plot(range(1, 151), epochs_150, 'r-', linewidth=2, label='150 Epochs (0.88)')
    ax.plot(range(1, 201), epochs_200, 'g-', linewidth=2, label='200 Epochs (0.92)')
    ax.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='Target 0.90')
    ax.set_title('Effect: Training Length 150 → 200 epochs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.suptitle('Individual Hyperparameter Effects on Training', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('hyperparameter_effects.png', dpi=150, bbox_inches='tight')
    print("✓ Hyperparameter effects plot saved to: hyperparameter_effects.png")
    plt.show()

if __name__ == "__main__":
    print("Generating comparison visualizations...\n")
    plot_training_comparison()
    print()
    plot_hyperparameter_effects()
    print("\n✓ All plots generated successfully!")
