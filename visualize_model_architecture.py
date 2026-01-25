"""
モデルアーキテクチャの可視化
StreamingConformer の構造を図示する
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config

def draw_architecture():
    fig, ax = plt.subplots(figsize=(14, 16))
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

    # Y座標（下から上へ）
    y_pos = 1.0

    def draw_box(y, text, color, height=box_height, width=box_width):
        box = FancyBboxPatch(
            (x_center - width/2, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x_center, y + height/2, text,
                ha='center', va='center', fontsize=11, weight='bold')
        return y + height

    def draw_arrow(y_from, y_to, label=''):
        arrow = FancyArrowPatch(
            (x_center, y_from), (x_center, y_to),
            arrowstyle='->',
            mutation_scale=20,
            linewidth=2,
            color='black'
        )
        ax.add_patch(arrow)
        if label:
            ax.text(x_center + 0.3, (y_from + y_to) / 2, label,
                    ha='left', va='center', fontsize=9, style='italic')

    def draw_residual_connection(y_start, y_end, x_offset=3.5):
        """残差接続を描画"""
        ax.annotate('',
                    xy=(x_center, y_end),
                    xytext=(x_center, y_start),
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle=f'arc3,rad=.5',
                                    lw=1.5,
                                    color='gray',
                                    linestyle='--'))

    # タイトル
    ax.text(x_center, 19, 'StreamingConformer Architecture',
            ha='center', va='center', fontsize=16, weight='bold')

    # 入力
    y_pos = draw_box(y_pos, f'Input: Mel Spectrogram\n(B, T, {config.N_MELS})', input_color)
    draw_arrow(y_pos, y_pos + 0.3)
    y_pos += 0.3

    # Subsampling
    y_pos = draw_box(y_pos, f'ConvSubsampling (2x downsample)\nConv2d(kernel=3, stride=2) + Linear\n→ (B, T/{config.SUBSAMPLING_RATE}, {config.D_MODEL})', subsample_color, height=1.0)
    draw_arrow(y_pos, y_pos + 0.3, f'×√{config.D_MODEL}')
    y_pos += 0.3

    # Positional Encoding
    y_pos = draw_box(y_pos, f'+ Relative Positional Encoding\n(d_model={config.D_MODEL})', subsample_color)
    draw_arrow(y_pos, y_pos + 0.3)
    y_pos += 0.3

    # Conformer Blocks
    for layer_idx in range(config.NUM_LAYERS):
        block_start_y = y_pos

        # Block label
        ax.text(x_center - box_width/2 - 0.5, y_pos + 2.5, f'Layer {layer_idx + 1}',
                ha='right', va='center', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor=encoder_color, alpha=0.7))

        # Layer Norm + FF1
        y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4)
        y_pos = draw_box(y_pos, f'Feed-Forward 1 (Macaron)\nLinear({config.D_MODEL} → {config.D_MODEL*4}) + SiLU + Dropout\nLinear({config.D_MODEL*4} → {config.D_MODEL})', ff_color, height=1.0)

        # Residual connection
        ax.text(x_center + box_width/2 + 0.3, y_pos - 0.5, '×0.5 + residual',
                ha='left', va='center', fontsize=8, style='italic', color='gray')

        # Layer Norm + Attention
        y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4)
        y_pos = draw_box(y_pos, f'Causal Multi-Head Self-Attention\n(heads={config.N_HEAD}, d_k={config.D_MODEL//config.N_HEAD})\nwith KV cache', attention_color, height=1.0)
        ax.text(x_center + box_width/2 + 0.3, y_pos - 0.5, '+ residual',
                ha='left', va='center', fontsize=8, style='italic', color='gray')

        # Conv Module (already has LayerNorm inside)
        y_pos = draw_box(y_pos, f'Conformer Conv Module\nLayerNorm + Pointwise Conv\nCausal DepthwiseConv (k={config.KERNEL_SIZE})\nPointwise Conv', conv_color, height=1.2)
        ax.text(x_center + box_width/2 + 0.3, y_pos - 0.6, '+ residual',
                ha='left', va='center', fontsize=8, style='italic', color='gray')

        # Layer Norm + FF2
        y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4)
        y_pos = draw_box(y_pos, f'Feed-Forward 2\nLinear({config.D_MODEL} → {config.D_MODEL*4}) + SiLU + Dropout\nLinear({config.D_MODEL*4} → {config.D_MODEL})', ff_color, height=1.0)
        ax.text(x_center + box_width/2 + 0.3, y_pos - 0.5, '×0.5 + residual',
                ha='left', va='center', fontsize=8, style='italic', color='gray')

        # Final LayerNorm
        y_pos = draw_box(y_pos, 'LayerNorm', encoder_color, height=0.4)

        draw_arrow(y_pos, y_pos + 0.3)
        y_pos += 0.3

    # Final layers
    y_pos = draw_box(y_pos, 'Dropout + LayerNorm', encoder_color)
    draw_arrow(y_pos, y_pos + 0.3)
    y_pos += 0.3

    # CTC Head
    y_pos = draw_box(y_pos, f'CTC Head\nLinear({config.D_MODEL} → {config.NUM_CLASSES})\n(including blank token)', output_color, height=0.8)
    draw_arrow(y_pos, y_pos + 0.3)
    y_pos += 0.3

    # Output
    y_pos = draw_box(y_pos, f'Output: Logits\n(B, T/{config.SUBSAMPLING_RATE}, {config.NUM_CLASSES})', output_color)

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
    • Mel Bins: {config.N_MELS}
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
