import matplotlib.pyplot as plt
import csv
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize training logs (history.csv)")
    parser.add_argument("--csv", type=str, default="checkpoints/history.csv", help="Path to history.csv")
    parser.add_argument("--output", type=str, default="diagnostics/training_curves.png", help="Output path for the plot")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found.")
        return

    epochs = []
    train_loss = []
    val_loss = []
    val_cer = []
    lrs = []

    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            val_loss.append(float(row['val_loss']))
            val_cer.append(float(row['val_cer']))
            lrs.append(float(row['lr']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot Loss
    ax1.plot(epochs, train_loss, label='Train Loss', marker='o')
    ax1.plot(epochs, val_loss, label='Val Loss', marker='x')
    ax1.set_ylabel('Loss (CTC)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log') # Loss often covers several orders of magnitude

    # Plot CER
    ax2.plot(epochs, val_cer, label='Val CER', color='red', marker='s')
    ax2.set_ylabel('Character Error Rate (CER)')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Validation CER')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(1.0, max(val_cer) if val_cer else 1.0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"Saved training curves to {args.output}")

if __name__ == "__main__":
    main()