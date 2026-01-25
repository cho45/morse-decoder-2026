import matplotlib.pyplot as plt
import csv
import argparse
import os

def plot_history(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    data = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    ep = int(row['epoch'])
                    data[ep] = {
                        'train_loss': float(row['train_loss']),
                        'val_loss': float(row['val_loss']),
                        'val_cer': float(row['val_cer']),
                        'cer_phrase': float(row.get('cer_phrase', 0)),
                        'cer_random': float(row.get('cer_random', 0)),
                        'lr': float(row['lr']),
                        'phase': int(row.get('phase', 0))
                    }
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping row {i+1} due to error: {e}")
                    continue
    except Exception as e:
        print(f"Error reading history csv: {e}")
        return

    if not data:
        print("No data to plot.")
        return

    # Sort by epoch to handle out-of-order logs from resume
    sorted_epochs = sorted(data.keys())
    epochs = sorted_epochs
    train_loss = [data[ep]['train_loss'] for ep in epochs]
    val_loss = [data[ep]['val_loss'] for ep in epochs]
    val_cer = [data[ep]['val_cer'] for ep in epochs]
    cer_phrase = [data[ep]['cer_phrase'] for ep in epochs]
    cer_random = [data[ep]['cer_random'] for ep in epochs]
    lrs = [data[ep]['lr'] for ep in epochs]
    phases = [data[ep]['phase'] for ep in epochs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Loss
    ax1.plot(epochs, train_loss, label='Train Loss', marker='o', markersize=4, alpha=0.7)
    ax1.plot(epochs, val_loss, label='Val Loss', marker='x', markersize=4, alpha=0.7)
    ax1.set_ylabel('Loss (CTC)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot CER
    ax2.plot(epochs, val_cer, label='Total CER', color='red', marker='s', markersize=4, linewidth=2)
    # Only plot phrase/random CER if they are non-zero (i.e. after they were introduced)
    if any(c > 0 for c in cer_phrase):
        ax2.plot(epochs, cer_phrase, label='Phrase CER', color='blue', marker='o', markersize=3, alpha=0.6, linestyle='--')
    if any(c > 0 for c in cer_random):
        ax2.plot(epochs, cer_random, label='Random CER', color='orange', marker='x', markersize=3, alpha=0.6, linestyle='--')
        
    ax2.set_ylabel('Character Error Rate (CER)')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Validation CER')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(2.0, max(0.5, max(val_cer) * 1.1 if val_cer else 1.0)))

    # Visualize Phases
    last_phase = -1
    for i, ep in enumerate(epochs):
        curr_phase = phases[i]
        if curr_phase != last_phase:
            # Phase changed
            ax1.axvline(x=ep, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=ep, color='gray', linestyle='--', alpha=0.5)
            # Use relative y-position for text to avoid being cut off
            ax1.text(ep, ax1.get_ylim()[1] * 0.9, f'P{curr_phase}', rotation=0, verticalalignment='top', fontweight='bold')
            last_phase = curr_phase

    # Add Learning Rate to ax1 with a twin axis
    ax1_lr = ax1.twinx()
    ax1_lr.plot(epochs, lrs, color='green', linestyle=':', label='LR', alpha=0.5)
    ax1_lr.set_ylabel('Learning Rate', color='green')
    ax1_lr.tick_params(axis='y', labelcolor='green')
    ax1_lr.set_yscale('log')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved training curves to {output_path}")
    plt.close(fig) # Close figure to free memory

def main():
    parser = argparse.ArgumentParser(description="Visualize training logs (history.csv)")
    parser.add_argument("--csv", type=str, default="checkpoints/history.csv", help="Path to history.csv")
    parser.add_argument("--output", type=str, default="diagnostics/training_curves.png", help="Output path for the plot")
    args = parser.parse_args()

    plot_history(args.csv, args.output)

if __name__ == "__main__":
    main()