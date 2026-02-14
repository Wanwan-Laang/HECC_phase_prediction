#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize HECC predictions from listing_comp_predictions.csv.
Requires: listing_comp.csv (for Ti/Zr/Nb/Ta fractions) and listing_comp_predictions.csv.
"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMP_CSV = os.path.join(SCRIPT_DIR, 'listing_comp.csv')
PRED_CSV = os.path.join(SCRIPT_DIR, 'listing_comp_predictions.csv')
OUT_DIR = SCRIPT_DIR


def ternary_to_cartesian(ti, zr, nb):
    """Convert (Ti, Zr, Nb) fractions (sum <= 1) to Cartesian for ternary plot.
    Vertices: Ti top, Zr bottom-left, Nb bottom-right."""
    s = ti + zr + nb
    if s <= 0:
        return np.nan, np.nan
    ti, zr, nb = ti / s, zr / s, nb / s
    x = zr + nb * 0.5
    y = nb * (3 ** 0.5) / 2
    return x, y


def load_merged():
    with open(COMP_CSV, newline='', encoding='utf-8') as f:
        comp = {r['composition_id']: r for r in csv.DictReader(f)}
    with open(PRED_CSV, newline='', encoding='utf-8') as f:
        pred = list(csv.DictReader(f))
    rows = []
    for r in pred:
        cid = r['composition_id']
        if cid not in comp:
            continue
        c = comp[cid]
        rows.append({
            'composition_id': cid,
            'Ti_frac': float(c['Ti_frac']),
            'Zr_frac': float(c['Zr_frac']),
            'Nb_frac': float(c['Nb_frac']),
            'Ta_frac': float(c['Ta_frac']),
            'ANN_multi_phase': float(r['ANN_multi_phase']),
            'SVM_multi_phase': float(r['SVM_multi_phase']),
        })
    return rows


def plot_ternary(rows, out_path, color_col='ANN_multi_phase', title_suffix='ANN'):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ti = np.array([r['Ti_frac'] for r in rows])
    zr = np.array([r['Zr_frac'] for r in rows])
    nb = np.array([r['Nb_frac'] for r in rows])
    colors = np.array([r[color_col] for r in rows])
    xs, ys = [], []
    for i in range(len(rows)):
        x, y = ternary_to_cartesian(ti[i], zr[i], nb[i])
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    valid = np.isfinite(xs) & np.isfinite(ys)
    sc = ax.scatter(xs[valid], ys[valid], c=colors[valid], s=18, cmap='viridis',
                    vmin=0, vmax=1, edgecolors='none')
    # Ternary triangle outline (Ti=1, Zr=1, Nb=1)
    ax.plot([0, 1, 0.5, 0], [0, 0, (3**0.5)/2, 0], 'k-', lw=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Ti–Zr–Nb (Ta = remainder)\ncolored by {title_suffix} multi-phase prob.')
    # Vertex labels
    ax.text(0, -0.08, 'Zr', ha='center', fontsize=11)
    ax.text(1, -0.08, 'Nb', ha='center', fontsize=11)
    ax.text(0.5, (3**0.5)/2 + 0.05, 'Ti', ha='center', fontsize=11)
    plt.colorbar(sc, ax=ax, shrink=0.7, label='P(multi-phase)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_ann_vs_svm(rows, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ann = [r['ANN_multi_phase'] for r in rows]
    svm = [r['SVM_multi_phase'] for r in rows]
    ax.scatter(ann, svm, s=12, alpha=0.7, c='#2e86ab', edgecolors='none')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='y = x')
    ax.set_xlabel('ANN multi-phase probability')
    ax.set_ylabel('SVM multi-phase probability')
    ax.set_title('ANN vs SVM predictions')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_ti_zr_scatter(rows, out_path):
    """2D scatter: Ti_frac vs Zr_frac, colored by ANN."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ti = [r['Ti_frac'] for r in rows]
    zr = [r['Zr_frac'] for r in rows]
    ann = [r['ANN_multi_phase'] for r in rows]
    sc = ax.scatter(ti, zr, c=ann, s=16, cmap='viridis', vmin=0, vmax=1, edgecolors='none')
    ax.set_xlabel('Ti fraction')
    ax.set_ylabel('Zr fraction')
    ax.set_title('Composition (Ti vs Zr)\ncolored by ANN multi-phase probability')
    plt.colorbar(sc, ax=ax, shrink=0.8, label='P(multi-phase)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.chdir(SCRIPT_DIR)
    if not os.path.isfile(PRED_CSV):
        print(f"Error: {PRED_CSV} not found. Run batch_predict.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(COMP_CSV):
        print(f"Error: {COMP_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    rows = load_merged()
    if not rows:
        print("No merged rows.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rows)} compositions.")

    plot_ternary(rows, os.path.join(OUT_DIR, 'ternary_ANN.png'),
                 color_col='ANN_multi_phase', title_suffix='ANN')
    plot_ternary(rows, os.path.join(OUT_DIR, 'ternary_SVM.png'),
                 color_col='SVM_multi_phase', title_suffix='SVM')
    plot_ann_vs_svm(rows, os.path.join(OUT_DIR, 'ANN_vs_SVM.png'))
    plot_ti_zr_scatter(rows, os.path.join(OUT_DIR, 'Ti_vs_Zr_ANN.png'))

    print("Done. Check: ternary_ANN.png, ternary_SVM.png, ANN_vs_SVM.png, Ti_vs_Zr_ANN.png")


if __name__ == '__main__':
    main()
