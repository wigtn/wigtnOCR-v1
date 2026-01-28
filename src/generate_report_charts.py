#!/usr/bin/env python3
"""
Generate charts for the Tech Report.

This script generates the following charts:
1. fig1_structure_f1_comparison.png - Baseline vs Advanced Structure F1
2. fig2_cer_comparison.png - Parser CER comparison
3. fig3_tradeoff_scatter.png - CER vs Structure F1 trade-off
4. fig4_latency_breakdown.png - Latency comparison
5. fig5_precision_recall.png - Precision-Recall for structure detection
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "tech_report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data from test_3 (Attention Is All You Need) - Primary benchmark
TEST_3_DATA = {
    "Text-Baseline": {
        "cer": 51.25,
        "wer": 57.19,
        "structure_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0, "fp": 11, "fn": 24,
        "latency": 2.31
    },
    "Image-Baseline": {
        "cer": 40.79,
        "wer": 41.24,
        "structure_f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0, "fp": 0, "fn": 24,
        "latency": 0.27
    },
    "Text-Advanced": {
        "cer": 64.11,
        "wer": 69.34,
        "structure_f1": 79.25,
        "precision": 72.41,
        "recall": 87.5,
        "tp": 21, "fp": 8, "fn": 3,
        "latency": 42.92
    },
    "Image-Advanced": {
        "cer": 57.71,
        "wer": 63.27,
        "structure_f1": 77.78,
        "precision": 70.0,
        "recall": 87.5,
        "tp": 21, "fp": 9, "fn": 3,
        "latency": 35.75
    }
}

# Colors for parsers
COLORS = {
    "Text-Baseline": "#3498db",
    "Image-Baseline": "#2ecc71",
    "Text-Advanced": "#e74c3c",
    "Image-Advanced": "#9b59b6"
}


def fig1_structure_f1_comparison():
    """Generate Structure F1 comparison chart (Baseline vs Advanced)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    parsers = list(TEST_3_DATA.keys())
    f1_scores = [TEST_3_DATA[p]["structure_f1"] for p in parsers]
    colors = [COLORS[p] for p in parsers]

    bars = ax.bar(parsers, f1_scores, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('Structure F1 Score (%)', fontsize=12)
    ax.set_title('Structure F1 Comparison: Baseline (0%) vs Advanced (~79%)\n(Test 3: Attention Is All You Need)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    # Add horizontal line at 0 for baseline reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add annotation for key insight
    ax.annotate('Advanced parsers achieve\n~79% Structure F1',
                xy=(2.5, 78), xytext=(1.5, 50),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_structure_f1_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig1_structure_f1_comparison.png")


def fig2_cer_comparison():
    """Generate CER comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    parsers = list(TEST_3_DATA.keys())
    cer_values = [TEST_3_DATA[p]["cer"] for p in parsers]
    colors = [COLORS[p] for p in parsers]

    bars = ax.bar(parsers, cer_values, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, cer in zip(bars, cer_values):
        height = bar.get_height()
        ax.annotate(f'{cer:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('Character Error Rate (%)', fontsize=12)
    ax.set_title('CER Comparison by Parser\n(Test 3: Attention Is All You Need)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)

    # Highlight lowest CER
    min_idx = cer_values.index(min(cer_values))
    bars[min_idx].set_edgecolor('gold')
    bars[min_idx].set_linewidth(3)

    # Add annotation
    ax.annotate('Lowest CER\n(Image-Baseline)',
                xy=(1, 40.79), xytext=(2, 25),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_cer_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig2_cer_comparison.png")


def fig3_tradeoff_scatter():
    """Generate CER vs Structure F1 trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for parser, data in TEST_3_DATA.items():
        ax.scatter(data["cer"], data["structure_f1"],
                   s=300, c=COLORS[parser], label=parser,
                   edgecolors='white', linewidth=2, alpha=0.8)

        # Add parser label
        offset_x = 2 if "Advanced" in parser else -2
        offset_y = 3 if "Text" in parser else -5
        ax.annotate(parser,
                    xy=(data["cer"], data["structure_f1"]),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    fontsize=10, ha='left' if "Advanced" in parser else 'right')

    ax.set_xlabel('Character Error Rate (%) - Lower is better', fontsize=12)
    ax.set_ylabel('Structure F1 Score (%) - Higher is better', fontsize=12)
    ax.set_title('Trade-off: CER vs Structure F1\n(Test 3: Attention Is All You Need)',
                 fontsize=14, fontweight='bold')

    # Add quadrant annotations
    ax.axhline(y=40, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=55, color='gray', linestyle='--', alpha=0.3)

    # Quadrant labels
    ax.text(45, 85, 'Ideal Zone\n(Low CER, High F1)', fontsize=9,
            ha='center', style='italic', color='green', alpha=0.7)
    ax.text(65, 5, 'Structure Loss\n(High CER, Low F1)', fontsize=9,
            ha='center', style='italic', color='red', alpha=0.7)

    ax.set_xlim(35, 75)
    ax.set_ylim(-5, 100)
    ax.legend(loc='center right', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_tradeoff_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig3_tradeoff_scatter.png")


def fig4_latency_breakdown():
    """Generate latency comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    parsers = list(TEST_3_DATA.keys())
    latencies = [TEST_3_DATA[p]["latency"] for p in parsers]
    colors = [COLORS[p] for p in parsers]

    bars = ax.bar(parsers, latencies, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, lat in zip(bars, latencies):
        height = bar.get_height()
        ax.annotate(f'{lat:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('Processing Time (seconds)', fontsize=12)
    ax.set_title('Latency Comparison by Parser\n(Test 3: Attention Is All You Need)',
                 fontsize=14, fontweight='bold')

    # Add speedup annotation
    baseline_fast = min(TEST_3_DATA["Text-Baseline"]["latency"], TEST_3_DATA["Image-Baseline"]["latency"])
    advanced_slow = max(TEST_3_DATA["Text-Advanced"]["latency"], TEST_3_DATA["Image-Advanced"]["latency"])
    speedup = advanced_slow / baseline_fast

    ax.annotate(f'Advanced is {speedup:.0f}x slower\nbut preserves structure',
                xy=(3, 40), xytext=(1.5, 35),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_latency_breakdown.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig4_latency_breakdown.png")


def fig5_precision_recall():
    """Generate Precision-Recall comparison for structure detection."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Only Advanced parsers have meaningful precision/recall
    parsers = ["Text-Advanced", "Image-Advanced"]
    x = np.arange(len(parsers))
    width = 0.35

    precision = [TEST_3_DATA[p]["precision"] for p in parsers]
    recall = [TEST_3_DATA[p]["recall"] for p in parsers]

    bars1 = ax.bar(x - width/2, precision, width, label='Precision',
                   color='#3498db', edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, recall, width, label='Recall',
                   color='#e74c3c', edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars1, precision):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    for bar, val in zip(bars2, recall):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Structure Detection: Precision vs Recall\n(Test 3: Attention Is All You Need)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(parsers)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=11)

    # Add TP/FP/FN counts
    for i, p in enumerate(parsers):
        tp, fp, fn = TEST_3_DATA[p]["tp"], TEST_3_DATA[p]["fp"], TEST_3_DATA[p]["fn"]
        ax.text(i, -8, f'TP:{tp} FP:{fp} FN:{fn}', ha='center', fontsize=9, color='gray')

    ax.set_ylim(-15, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_precision_recall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig5_precision_recall.png")


def main():
    """Generate all charts."""
    print("Generating Tech Report Charts...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    fig1_structure_f1_comparison()
    fig2_cer_comparison()
    fig3_tradeoff_scatter()
    fig4_latency_breakdown()
    fig5_precision_recall()

    print("-" * 50)
    print("All charts generated successfully!")


if __name__ == "__main__":
    main()
