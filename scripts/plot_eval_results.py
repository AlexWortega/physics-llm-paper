"""
Generate paper figures from eval_data/ outputs:
  - figures/rollout_mse_curve.pdf   — MSE(t) for 4 scenarios
  - figures/conservation_drift.pdf  — momentum + KE drift
  - figures/collision_decomp.pdf    — collision vs free-flight bar chart
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

EVAL = Path('/home/alexw/Projects/physics-llm-paper/eval_data')
FIGS = Path('/home/alexw/Projects/physics-llm-paper/figures')
FIGS.mkdir(exist_ok=True)

COLORS = {
    'Constraint':  '#2196F3',
    'Stacking':    '#FF9800',
    'Collision':   '#4CAF50',
    'OOD-novel':   '#F44336',
}

def plot_rollout():
    data = json.loads((EVAL / 'rollout_mse.json').read_text())

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for scen, row in data.items():
        curve = np.array(row['mean_mse_curve'])
        rmse = np.sqrt(np.maximum(curve, 0))
        cat = row['category']
        label = f"{scen.replace('_',' ').title()} ({cat})"
        ax.plot(range(1, len(rmse)+1), rmse,
                color=COLORS.get(cat, '#888'), label=label, linewidth=1.8)

    ax.set_xlabel('Rollout step $t$')
    ax.set_ylabel('Avg RMSE (px)')
    ax.set_title('Multi-step rollout error', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_xlim(1, max(len(v['mean_mse_curve']) for v in data.values()))
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'rollout_mse_curve.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved rollout_mse_curve.pdf/.png")


def plot_conservation():
    p = EVAL / 'conservation.json'
    if not p.exists():
        print("conservation.json not found, skipping")
        return
    data = json.loads(p.read_text())
    p_drift = np.array(data['momentum_drift_curve'])
    ke_drift = np.array(data['ke_drift_curve'])
    steps = range(1, len(p_drift)+1)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].plot(steps, p_drift * 100, color='#2196F3', linewidth=1.8)
    axes[0].set_xlabel('Rollout step'); axes[0].set_ylabel('Momentum drift (%)')
    axes[0].set_title('Momentum conservation', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, ke_drift * 100, color='#F44336', linewidth=1.8)
    axes[1].set_xlabel('Rollout step'); axes[1].set_ylabel('KE drift (%)')
    axes[1].set_title('Kinetic energy conservation', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'conservation_drift.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved conservation_drift.pdf/.png")


def plot_collision_decomp():
    data = json.loads((EVAL / 'collision_decomp.json').read_text())
    cats = data['per_category']

    CAT_ORDER = ['Collision', 'Stacking', 'Ramp', 'Constraint', 'Minigame', 'Complex']
    cats_present = [c for c in CAT_ORDER if c in cats]

    col_frac  = [cats[c]['col_frac'] * 100 for c in cats_present]
    col_lin   = [cats[c]['col_lin_mse'] for c in cats_present]
    flight_lin = [cats[c]['flight_lin_mse'] for c in cats_present]

    x = np.arange(len(cats_present))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    axes[0].bar(x, col_frac, color='#F44336', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats_present, fontsize=8)
    axes[0].set_ylabel('Collision frames (%)'); axes[0].set_title('Collision frame fraction', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x - w/2, col_lin,   width=w, label='Collision', color='#F44336', alpha=0.8)
    axes[1].bar(x + w/2, flight_lin, width=w, label='Free-flight', color='#4CAF50', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats_present, fontsize=8)
    axes[1].set_ylabel('Lin. extrap MSE (px²)'); axes[1].set_title('Difficulty: collision vs free-flight', fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'collision_decomp.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved collision_decomp.pdf/.png")


if __name__ == '__main__':
    plot_rollout()
    plot_conservation()
    plot_collision_decomp()
    print("Done — all figures written to", FIGS)
