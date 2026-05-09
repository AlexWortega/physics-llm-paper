"""
Generate paper figures from eval_data/ outputs:
  - figures/rollout_mse_curve.pdf   — MSE(t) with std shading for 4 scenarios
  - figures/conservation_drift.pdf  — horizontal px error + KE error (in-distribution)
  - figures/collision_decomp.pdf    — collision vs free-flight bar chart + quantitative table
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

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for scen, row in data.items():
        curve = np.array(row['mean_mse_curve'])
        rmse  = np.sqrt(np.maximum(curve, 0))
        cat   = row['category']
        label = f"{scen.replace('_',' ').title()} ({cat})"
        color = COLORS.get(cat, '#888')
        steps = np.arange(1, len(rmse) + 1)
        ax.plot(steps, rmse, color=color, label=label, linewidth=1.8)

        # std shading (convert MSE std to RMSE std via √(MSE+std) - √MSE approx)
        if 'std_mse_curve' in row:
            std  = np.array(row['std_mse_curve'])
            # use per-scene RMSE std if per_scene_curves available
            if 'per_scene_curves' in row:
                per_sc = np.array(row['per_scene_curves'])
                per_rmse = np.sqrt(np.maximum(per_sc, 0))
                rmse_std = np.nanstd(per_rmse, axis=0)
            else:
                rmse_std = std / (2 * rmse + 1e-6)  # delta method
            ax.fill_between(steps,
                            np.maximum(rmse - rmse_std, 0),
                            rmse + rmse_std,
                            color=color, alpha=0.15)

    ax.set_xlabel('Rollout step $t$', fontsize=10)
    ax.set_ylabel('Avg RMSE (px)', fontsize=10)
    ax.set_title('Multi-step autoregressive rollout error', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
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

    px_curve = np.array(data['px_err_curve'])
    px_std   = np.array(data.get('px_err_std_curve', np.zeros_like(px_curve)))
    mean_ke  = data.get('mean_ke_err_free_flight', None)
    std_ke   = data.get('std_ke_err_free_flight', 0.0)
    steps    = np.arange(1, len(px_curve) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # Left: horizontal momentum error curve
    axes[0].plot(steps, px_curve * 100, color='#2196F3', linewidth=1.8,
                 label='mean px error')
    axes[0].fill_between(steps,
                         np.maximum((px_curve - px_std) * 100, 0),
                         (px_curve + px_std) * 100,
                         color='#2196F3', alpha=0.15)
    axes[0].set_xlabel('Rollout step', fontsize=10)
    axes[0].set_ylabel('Horizontal momentum error (%)', fontsize=9)
    axes[0].set_title('Horizontal momentum\n(gravity-free axis)', fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)

    # Right: KE error (scalar, show as bar with error bar)
    if mean_ke is not None:
        axes[1].bar(['Free-flight KE\nerror'],
                    [mean_ke * 100], yerr=[std_ke * 100],
                    color='#F44336', alpha=0.75, width=0.4, capsize=8)
        axes[1].set_ylabel('|KE_pred − KE_gt| / KE_gt (%)', fontsize=9)
        axes[1].set_title('Kinetic energy error\n(free-flight frames only)', fontsize=9)
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3, axis='y')

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

    col_frac   = [cats[c]['col_frac'] * 100 for c in cats_present]
    col_lin    = [cats[c]['col_lin_mse'] for c in cats_present]
    flight_lin = [cats[c]['flight_lin_mse'] for c in cats_present]

    x = np.arange(len(cats_present))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    # Left: fraction of collision frames per category
    axes[0].bar(x, col_frac, color='#F44336', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats_present, fontsize=8)
    axes[0].set_ylabel('Collision frames (%)', fontsize=9)
    axes[0].set_title('Fraction of collision frames\nper category', fontsize=9)
    axes[0].set_ylim(0, 110)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: linear extrap MSE on collision vs free-flight frames
    axes[1].bar(x - w/2, col_lin,    width=w, label='Collision frames', color='#F44336', alpha=0.8)
    axes[1].bar(x + w/2, flight_lin, width=w, label='Free-flight frames', color='#4CAF50', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats_present, fontsize=8)
    axes[1].set_ylabel('Linear extrap MSE (px²)', fontsize=9)
    axes[1].set_title('Prediction difficulty:\ncollision vs. free-flight', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')

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
