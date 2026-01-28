import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import config

def visualize_fixture():
    fixture_path = "tests/fixtures/decoding_test.json"
    with open(fixture_path, "r") as f:
        fixtures = json.load(f)

    case = fixtures["fast_40wpm"]
    logits = torch.tensor(case["logits"])
    sig_logits = torch.tensor(case["sig_logits"])
    bound_logits = torch.tensor(case["bound_logits"])
    bound_probs = torch.sigmoid(bound_logits).squeeze(-1)
    
    ctc_probs = torch.softmax(logits, dim=-1).numpy()
    sig_preds = torch.softmax(sig_logits, dim=-1).numpy()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # 1. CTC Spikes
    # Plot probability of non-blank characters
    char_probs = 1.0 - ctc_probs[:, 0]
    axes[0].plot(char_probs, label="Char Prob (Sum)", color='blue')
    axes[0].set_title("CTC Character Probabilities")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    
    # Annotate spikes
    spikes = ctc_probs.argmax(axis=-1)
    for t, idx in enumerate(spikes):
        if idx != 0:
            char = config.ID_TO_CHAR.get(idx, "?")
            axes[0].annotate(char, (t, char_probs[t]), xytext=(0, 10), textcoords='offset points', ha='center', color='red', weight='bold')

    # 2. Signal Head
    # 0:bg, 1:dit, 2:dah, 3:word
    colors = ['gray', 'green', 'red', 'orange']
    labels = ['Background', 'Dit', 'Dah', 'Word Space']
    for i in range(1, 4):
        axes[1].plot(sig_preds[:, i], label=labels[i], color=colors[i], alpha=0.7)
    
    axes[1].set_title("Signal Head Predictions")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right', fontsize='small')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Boundary Head
    axes[2].plot(bound_probs.numpy(), label="Boundary Prob", color='purple')
    axes[2].set_title("Boundary Head Predictions")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc='upper right', fontsize='small')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "fixture_visualization.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_fixture()