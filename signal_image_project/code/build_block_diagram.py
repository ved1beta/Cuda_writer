"""Draw a system block diagram of the codec for use in slides/report."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent.parent / "outputs" / "fig_block_diagram.png"


def box(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor="#1f2a44", facecolor=color))
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=11, color="#1f2a44")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=4,head_length=6",
        linewidth=1.6, color="#1f2a44",
        shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    encoder = "#dbeafe"
    decoder = "#fde68a"

    ax.text(6, 4.25, "JPEG-baseline codec pipeline",
            ha="center", va="center", fontsize=14, weight="bold",
            color="#1f2a44")

    box(ax, 0.0, 2.6, 1.6, 0.9, "Input\nimage", "#ffffff")
    box(ax, 1.9, 2.6, 1.6, 0.9, "Level shift\n(-128)", encoder)
    box(ax, 3.8, 2.6, 1.6, 0.9, "8x8\nblocking", encoder)
    box(ax, 5.7, 2.6, 1.6, 0.9, "2D DCT\nF = T f Tᵀ", encoder)
    box(ax, 7.6, 2.6, 1.6, 0.9, "Quantise\nround(F / Q')", encoder)
    box(ax, 9.5, 2.6, 2.3, 0.9, "Entropy estimate\n(Shannon bound)", encoder)

    for x1, x2 in [(1.6, 1.9), (3.5, 3.8), (5.4, 5.7), (7.3, 7.6), (9.2, 9.5)]:
        arrow(ax, x1, 3.05, x2, 3.05)

    arrow(ax, 10.65, 2.55, 10.65, 1.95)

    box(ax, 9.5, 1.0, 2.3, 0.9, "Dequantise\nF * Q'", decoder)
    box(ax, 7.6, 1.0, 1.6, 0.9, "2D IDCT\nf = Tᵀ F T", decoder)
    box(ax, 5.7, 1.0, 1.6, 0.9, "Un-shift\n(+128)", decoder)
    box(ax, 3.8, 1.0, 1.6, 0.9, "Clip\n[0, 255]", decoder)
    box(ax, 1.9, 1.0, 1.6, 0.9, "Crop\npadding", decoder)
    box(ax, 0.0, 1.0, 1.6, 0.9, "Output\nimage", "#ffffff")

    for x1, x2 in [(9.5, 9.2), (7.6, 7.3), (5.7, 5.4), (3.8, 3.5), (1.9, 1.6)]:
        arrow(ax, x1, 1.45, x2, 1.45)

    ax.text(0.2, 0.2, "encoder", color="#1e3a8a", fontsize=10)
    ax.text(0.2, 0.05, "decoder", color="#92400e", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
