"""
モデルアーキテクチャの可視化
StreamingConformer の構造を図示する
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config

def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # 色定義
    input_color = '#E8F4F8'
    subsample_color = '#B8E6F0'
    encoder_color = '#FFE5B4'
    attention_color = '#FFD4A3'
    conv_color = '#C7E9C0'
    ff_color = '#FFDAB9'
    output_color = '#F0E68C'

    # ボックスの幅と高さ
    box_width = 6
    box_height = 0.6
    x_center = 5

    # Y座標（上から下へ）
    y_pos = 18.5

    def draw_box(y, text, color, height=box_height, width=box_width, x=None, fontsize=10):
        if x is None:
            x = x_center
        box = FancyBboxPatch(
            (x - width/2, y - height), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, y - height/2, text,
                ha='center', va='center', fontsize=fontsize, weight='bold')
        return y - height

    def draw_arrow(y_from, y_to, label='', x=None):
        if x is None:
            x = x_center
        arrow = FancyArrowPatch(
            (x, y_from), (x, y_to),
            arrowstyle='->',
            mutation_scale=20,
            linewidth=2,
            color='black'
        )
        ax.add_patch(arrow)
        if label:
            ax.text(x + 0.3, (y_from + y_to) / 2, label,
                    ha='left', va='center', fontsize=9, style='italic')

    # タイトル
    ax.text(x_center, 19.5, 'StreamingConformer Architecture',
            ha='center', va='center', fontsize=16, weight='bold')

    # 入力
    y_pos = draw_box(y_pos, f'Input: Linear Spectrogram\n(B, T, {config.N_BINS})', input_color)
    draw_arrow(y_pos, y_pos - 0.4)
    y_pos -= 0.4

    # Subsampling
    y_pos = draw_box(y_pos, f'ConvSubsampling (2x downsample)\nConv2d(kernel=3, stride=2) + Linear\n→ (B, T/{config.SUBSAMPLING_RATE}, {config.D_MODEL})', subsample_color, height=1.0)
    draw_arrow(y_pos, y_pos - 0.4, f'×√{config.D_MODEL}')
    y_pos -= 0.4

    # Positional Encoding
    y_pos = draw_box(y_pos, f'+ Relative Positional Encoding\n(d_model={config.D_MODEL})', subsample_color)
    draw_arrow(y_pos, y_pos - 0.4)
    y_pos -= 0.4

    # Conformer Blocks (Represented as a single unit)
    block_top = y_pos
    
    # Outer boundary for the block
    block_height = 6.0
    rect = FancyBboxPatch(
        (x_center - box_width/2 - 0.5, y_pos - block_height), box_width + 1.0, block_height,
        boxstyle="round,pad=0.2", edgecolor='gray', facecolor='none', linestyle='--', linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x_center + box_width/2 + 0.8, y_pos - block_height/2, f'× {config.NUM_LAYERS} Layers',
            ha='left', va='center', fontsize=14, weight='bold', color='red')

    # Inside the block
    y_pos -= 0.2
    y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4, width=box_width-0.5)
    y_pos = draw_box(y_pos, f'Feed-Forward 1 (Macaron)\nLinear({config.D_MODEL} → {config.D_MODEL*4}) + SiLU + Dropout\nLinear({config.D_MODEL*4} → {config.D_MODEL})', ff_color, height=1.0, width=box_width-0.5, fontsize=9)
    ax.text(x_center + box_width/2 - 0.5, y_pos + 0.5, '×0.5 + residual', ha='right', va='center', fontsize=8, style='italic', color='gray')

    y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4, width=box_width-0.5)
    y_pos = draw_box(y_pos, f'Causal Multi-Head Self-Attention\n(heads={config.N_HEAD}, d_k={config.D_MODEL//config.N_HEAD}) with KV cache', attention_color, height=1.0, width=box_width-0.5, fontsize=9)
    ax.text(x_center + box_width/2 - 0.5, y_pos + 0.5, '+ residual', ha='right', va='center', fontsize=8, style='italic', color='gray')

    y_pos = draw_box(y_pos, f'Conformer Conv Module\nLayerNorm + Pointwise Conv + GLU\nCausal DepthwiseConv (k={config.KERNEL_SIZE})\nLayerNorm + SiLU + Pointwise Conv', conv_color, height=1.2, width=box_width-0.5, fontsize=9)
    ax.text(x_center + box_width/2 - 0.5, y_pos + 0.6, '+ residual', ha='right', va='center', fontsize=8, style='italic', color='gray')

    y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4, width=box_width-0.5)
    y_pos = draw_box(y_pos, f'Feed-Forward 2\nLinear({config.D_MODEL} → {config.D_MODEL*4}) + SiLU + Dropout\nLinear({config.D_MODEL*4} → {config.D_MODEL})', ff_color, height=1.0, width=box_width-0.5, fontsize=9)
    ax.text(x_center + box_width/2 - 0.5, y_pos + 0.5, '×0.5 + residual', ha='right', va='center', fontsize=8, style='italic', color='gray')

    y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4, width=box_width-0.5)
    
    y_pos = block_top - block_height
    draw_arrow(y_pos, y_pos - 0.6)
    y_pos -= 0.6

    # Final layers
    y_pos = draw_box(y_pos, 'Dropout + LayerNorm', encoder_color)
    
    # Branching to 3 heads
    branch_y = y_pos
    draw_arrow(branch_y, branch_y - 0.6, x=x_center)      # To CTC Head
    draw_arrow(branch_y, branch_y - 0.6, x=x_center-3)    # To Signal Head
    draw_arrow(branch_y, branch_y - 0.6, x=x_center+3)    # To Boundary Head
    y_pos -= 0.6

    # 3 Heads
    head_width = 2.5
    draw_box(y_pos, f'Signal Head\nLinear({config.D_MODEL} → {config.NUM_SIGNAL_CLASSES})', output_color, height=0.8, width=head_width, x=x_center-3)
    draw_box(y_pos, f'CTC Head\nLinear({config.D_MODEL} → {config.NUM_CLASSES})\n(incl. blank)', output_color, height=0.8, width=head_width, x=x_center)
    draw_box(y_pos, f'Boundary Head\nLinear({config.D_MODEL} → 1)', output_color, height=0.8, width=head_width, x=x_center+3)
    
    y_pos -= 0.8
    draw_arrow(y_pos, y_pos - 0.5, x=x_center-3)
    draw_arrow(y_pos, y_pos - 0.5, x=x_center)
    draw_arrow(y_pos, y_pos - 0.5, x=x_center+3)
    y_pos -= 0.5

    # Outputs
    draw_box(y_pos, 'Signal Logits', output_color, height=0.6, width=head_width, x=x_center-3)
    draw_box(y_pos, 'CTC Logits', output_color, height=0.6, width=head_width, x=x_center)
    draw_box(y_pos, 'Boundary Logits', output_color, height=0.6, width=head_width, x=x_center+3)

    # 凡例を追加
    legend_y = 0.5
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input/Output'),
        mpatches.Patch(facecolor=subsample_color, edgecolor='black', label='Preprocessing'),
        mpatches.Patch(facecolor=attention_color, edgecolor='black', label='Attention'),
        mpatches.Patch(facecolor=conv_color, edgecolor='black', label='Convolution'),
        mpatches.Patch(facecolor=ff_color, edgecolor='black', label='Feed-Forward'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

    # モデル設定情報を追加
    info_text = f"""Model Configuration:
    • Sample Rate: {config.SAMPLE_RATE} Hz
    • Frequency Bins: {config.N_BINS}
    • d_model: {config.D_MODEL}
    • Layers: {config.NUM_LAYERS}
    • Attention Heads: {config.N_HEAD}
    • Conv Kernel: {config.KERNEL_SIZE}
    • Dropout: {config.DROPOUT}
    • Vocabulary: {config.NUM_CLASSES} classes"""

    ax.text(0.5, 17, info_text,
            ha='left', va='top', fontsize=8,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = draw_architecture()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    print("Model architecture diagram saved to model_architecture.png")
    plt.show()
