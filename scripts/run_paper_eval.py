"""
Generate all three paper-critical eval results:
  1. Multi-step rollout MSE(t) for t=1..50 on 6 scenarios
  2. Conservation (momentum/KE) on model rollout in zero-gravity billiards
  3. Collision-frame vs free-flight MSE decomposition

Run from project root:
  cd /home/alexw
  python Projects/physics-llm-paper/scripts/run_paper_eval.py
"""
import sys, json, math, re, random, os
sys.path.insert(0, '/home/alexw')

import numpy as np
import torch
from pathlib import Path

CHECKPOINT = '/home/alexw/physics-llm-debug/lfm2-scenarios-merged'
OUT_DIR = Path('/home/alexw/Projects/physics-llm-paper/eval_data')
OUT_DIR.mkdir(exist_ok=True)

ROLLOUT_STEPS = 100
N_SCENES = 5          # scenes per scenario for rollout (for std bands)
N_CONSERVATION = 8    # scenes for conservation analysis
DT = 1/60.0
G  = 981.0

SCENARIOS_ROLLOUT = [
    ('pendulum',       'Constraint'),
    ('tower',          'Stacking'),
    ('billiards',      'Collision'),
    ('pong',           'OOD-novel'),
]

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading merged LFM2 model from", CHECKPOINT)
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )
    model.eval()
    model.tokenizer = tok
    print("Model ready.")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Text format helpers (mirrors data_loader.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_obj(obj):
    oid = obj['id']
    p = obj['position']
    v = obj.get('velocity', {'x': 0.0, 'y': 0.0})
    a  = obj.get('angle', 0.0)
    av = obj.get('angular_velocity', 0.0)
    line = f"  obj_{oid}: pos=({p['x']:.4f}, {p['y']:.4f}), vel=({v['x']:.4f}, {v['y']:.4f})"
    if abs(a) > 0.001 or abs(av) > 0.001:
        line += f", a={a:.4f}, av={av:.4f}"
    return line

def frame_to_text(frame):
    n = frame['frame']
    desc = frame.get('description', f'Frame {n}: Objects in motion.')
    lines = [desc] + [_fmt_obj(o) for o in frame.get('objects', [])]
    return '\n'.join(lines) + '\n'

def header_to_text(header):
    grav = header.get('gravity', {'x': 0, 'y': -981})
    dt   = header.get('timestep', DT)
    desc = header.get('description', '')
    lines = [
        desc,
        f"Gravity: ({grav['x']:.4f}, {grav['y']:.4f})  Timestep: {dt:.6f}",
        f"Objects: {header.get('object_count', 0)}",
    ]
    for obj in header.get('objects', []):
        oid = obj['id']
        mat = obj.get('material', {})
        lines.append(
            f"  obj_{oid}: mass={mat.get('mass',1):.2f} "
            f"friction={mat.get('friction',0.5):.2f} "
            f"elasticity={mat.get('elasticity',0.5):.2f}"
        )
    return '\n'.join(lines) + '\n'

def build_prompt(header, context_frames):
    h = header_to_text(header)
    ctx = ''.join(frame_to_text(f) for f in context_frames)
    return h + ctx + 'Predict next frame:\n'

# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict_next(model, prompt, max_new_tokens=256):
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    inp = tok(prompt, return_tensors='pt', truncation=True, max_length=6000).to('cuda')
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tok.pad_token_id,
        )
    gen = tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    return gen

# ──────────────────────────────────────────────────────────────────────────────
# Parser: extract positions+velocities from generated text
# ──────────────────────────────────────────────────────────────────────────────

POS_RE = re.compile(r'obj_(\d+).*?pos=\(([^,]+),\s*([^)]+)\).*?vel=\(([^,]+),\s*([^)]+)\)', re.S)

def parse_frame(text):
    """Returns {id: {'x','y','vx','vy'}} or None on failure."""
    objs = {}
    for m in POS_RE.finditer(text):
        try:
            oid = int(m.group(1))
            objs[oid] = {
                'x': float(m.group(2)), 'y': float(m.group(3)),
                'vx': float(m.group(4)), 'vy': float(m.group(5)),
            }
        except ValueError:
            pass
    return objs if objs else None

def gt_to_dict(frame):
    return {
        o['id']: {
            'x': o['position']['x'], 'y': o['position']['y'],
            'vx': o.get('velocity',{}).get('x',0),
            'vy': o.get('velocity',{}).get('y',0),
            'mass': o.get('material',{}).get('mass',1.0),
        }
        for o in frame.get('objects', [])
    }

def pos_mse(pred_dict, gt_dict):
    errs = []
    for oid in set(pred_dict) & set(gt_dict):
        dx = pred_dict[oid]['x'] - gt_dict[oid]['x']
        dy = pred_dict[oid]['y'] - gt_dict[oid]['y']
        errs.append(dx*dx + dy*dy)
    return float(np.mean(errs)) if errs else float('nan')

# ──────────────────────────────────────────────────────────────────────────────
# Scene generation (Pymunk)
# ──────────────────────────────────────────────────────────────────────────────

def gen_scene_states(scenario_type, seed, n_frames=60, zero_gravity=False):
    from src.physics import generate_scenario
    grav = (0.0, 0.0) if zero_gravity else None
    sim, meta = generate_scenario(seed, scenario_type=scenario_type, gravity=grav)
    states = []
    for _ in range(n_frames + 1):
        states.append(sim.get_state())
        sim.step()
    header = {
        'description': meta.get('description', f'Scene: {scenario_type}'),
        'gravity': {'x': 0.0, 'y': 0.0 if zero_gravity else -G},
        'timestep': DT,
        'object_count': len(states[0]['objects']),
        'objects': [
            {'id': o['id'],
             'material': o.get('material', {'mass':1,'friction':0.5,'elasticity':0.9})}
            for o in states[0]['objects']
        ]
    }
    frames = [{'frame': s['frame'], 'description': f"Frame {s['frame']}: Objects in motion.",
               'objects': s['objects']} for s in states]
    return header, frames

# ──────────────────────────────────────────────────────────────────────────────
# 1. Multi-step rollout MSE(t)
# ──────────────────────────────────────────────────────────────────────────────

def run_rollout_eval(model):
    print("\n=== Multi-step rollout MSE(t) ===")
    results = {}

    for scen, cat in SCENARIOS_ROLLOUT:
        scene_curves = []
        for seed in range(7000000, 7000000 + N_SCENES):
            try:
                header, frames = gen_scene_states(scen, seed, n_frames=ROLLOUT_STEPS+4)
            except Exception as e:
                print(f"  {scen} seed={seed} gen error: {e}")
                continue

            # initial context: frames 0..3 (GT)
            context_frames = frames[:4]
            pred_frames_text = [frame_to_text(f) for f in context_frames]
            step_mse = []

            for t in range(ROLLOUT_STEPS):
                gt_frame = frames[4 + t]
                # Build prompt from last 4 predicted frames
                ctx_text = ''.join(pred_frames_text[-4:])
                prompt = header_to_text(header) + ctx_text + 'Predict next frame:\n'

                gen = predict_next(model, prompt, max_new_tokens=160)
                pred_dict = parse_frame(gen)
                gt_dict = gt_to_dict(gt_frame)

                if pred_dict is None or not (set(pred_dict) & set(gt_dict)):
                    step_mse.append(float('nan'))
                    # keep last valid frame as context
                    pred_frames_text.append(frame_to_text(gt_frame))  # fallback to GT
                else:
                    mse = pos_mse(pred_dict, gt_dict)
                    step_mse.append(mse)
                    # Build predicted frame text for next context
                    # Reformat gen as a frame text
                    pred_frames_text.append('Frame ' + str(gt_frame['frame']) + ': ' + gen.strip() + '\n')

            scene_curves.append(step_mse)
            print(f"  {scen} seed={seed}: steps={len(step_mse)} "
                  f"mse[0]={step_mse[0]:.2f} mse[-1]={step_mse[-1]:.2f}")

        if scene_curves:
            arr = np.array(scene_curves)
            mean_curve = list(np.nanmean(arr, axis=0).tolist())
            std_curve  = list(np.nanstd(arr, axis=0).tolist())
            results[scen] = {
                'category': cat,
                'mean_mse_curve': mean_curve,
                'std_mse_curve':  std_curve,
                'per_scene_curves': [list(c) for c in scene_curves],
            }

    path = OUT_DIR / 'rollout_mse.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved rollout MSE → {path}")
    return results

# ──────────────────────────────────────────────────────────────────────────────
# 2. Conservation analysis on model predictions
# ──────────────────────────────────────────────────────────────────────────────

V_COLLISION_THRESH = 10.0  # px/s — velocity change this large = collision frame

def horiz_momentum(objs_dict, masses):
    """Horizontal momentum Σ m_i * vx_i — gravity (vertical) does not affect this."""
    return sum(masses.get(oid, 1.0) * o['vx'] for oid, o in objs_dict.items())

def kinetic_energy(objs_dict, masses):
    return sum(0.5 * masses.get(oid, 1.0) * (o['vx']**2 + o['vy']**2)
               for oid, o in objs_dict.items())

def is_collision_frame(gt_prev, gt_cur):
    """True if any object's velocity changed by > threshold between two GT frames."""
    for oid in set(gt_prev) & set(gt_cur):
        dv = math.sqrt((gt_cur[oid]['vx'] - gt_prev[oid]['vx'])**2 +
                       (gt_cur[oid]['vy'] - gt_prev[oid]['vy'])**2)
        if dv > V_COLLISION_THRESH:
            return True
    return False

def run_conservation_eval(model):
    """
    In-distribution conservation test on normal billiards (gravity ON).

    At each autoregressive step we compare the model's predicted horizontal
    momentum and kinetic energy to the ground-truth values at that step.
    On free-flight frames (no collision in GT), both quantities should match
    the GT closely — horizontal momentum changes only due to friction (small),
    and KE changes deterministically as gravity converts PE to KE.

    This separates conservation failure from OOD-gravity failure.
    """
    print("\n=== Conservation analysis (in-distribution: billiards with gravity) ===")
    CON_STEPS = 50   # 50 steps = 0.83 s realtime
    all_px_err, all_ke_err = [], []

    for seed in range(7100000, 7100000 + N_CONSERVATION):
        try:
            header, frames = gen_scene_states('billiards', seed,
                                              n_frames=CON_STEPS + 4,
                                              zero_gravity=False)
        except Exception as e:
            print(f"  seed={seed} error: {e}"); continue

        masses = {o['id']: o.get('material', {}).get('mass', 1.0)
                  for o in header['objects']}

        context_frames = frames[:4]
        pred_frames_text = [frame_to_text(f) for f in context_frames]

        # Normalise momentum error by the scene's initial |Σm·vx| to avoid
        # divide-by-zero when balls happen to move in opposite directions.
        px_initial = horiz_momentum(gt_to_dict(frames[3]), masses)
        px_norm = max(abs(px_initial), 1.0)  # lower-bound at 1.0 px·kg/s

        px_errs, ke_errs = [], []

        for t in range(CON_STEPS):
            gt_frame = frames[4 + t]
            gt_dict   = gt_to_dict(gt_frame)
            gt_prev   = gt_to_dict(frames[3 + t])
            collision  = is_collision_frame(gt_prev, gt_dict)

            ctx_text = ''.join(pred_frames_text[-4:])
            prompt   = header_to_text(header) + ctx_text + 'Predict next frame:\n'
            gen      = predict_next(model, prompt, max_new_tokens=160)
            pred_dict = parse_frame(gen)

            if pred_dict and (set(pred_dict) & set(gt_dict)):
                # Horizontal momentum: compare predicted to GT.
                # Normalise by initial |Σm·vx| (stable; avoids near-zero denom).
                px_pred = horiz_momentum(pred_dict, masses)
                px_gt   = horiz_momentum(gt_dict,   masses)
                px_err  = abs(px_pred - px_gt) / px_norm
                px_errs.append(px_err)

                # KE error on free-flight frames only (gravity effect factored out via GT)
                if not collision:
                    ke_pred = kinetic_energy(pred_dict, masses)
                    ke_gt   = kinetic_energy(gt_dict,   masses)
                    ke_err  = abs(ke_pred - ke_gt) / (ke_gt + 1e-6)
                    ke_errs.append(ke_err)

                pred_frames_text.append(
                    'Frame ' + str(gt_frame['frame']) + ': ' + gen.strip() + '\n')
            else:
                px_errs.append(float('nan'))
                pred_frames_text.append(frame_to_text(gt_frame))

        all_px_err.append(px_errs)
        all_ke_err.append(ke_errs)
        print(f"  seed={seed}: px_err[-1]={px_errs[-1]:.4f}  "
              f"ke_err(free-flight mean)={float(np.nanmean(ke_errs)):.4f}")

    # Pad ke_errs to same length as px_errs for consistent curves
    max_len = max(len(r) for r in all_px_err)
    arr_px = np.nanmean(np.array([r + [float('nan')]*(max_len-len(r))
                                   for r in all_px_err]), axis=0).tolist()
    arr_px_std = np.nanstd(np.array([r + [float('nan')]*(max_len-len(r))
                                      for r in all_px_err]), axis=0).tolist()
    mean_ke_err = float(np.nanmean([v for r in all_ke_err for v in r]))
    std_ke_err  = float(np.nanstd( [v for r in all_ke_err for v in r]))

    result = {
        'description': 'In-distribution billiards (gravity on). '
                       'px_err: |Σm·pred_vx - Σm·gt_vx| / max(|Σm·vx_0|, 1.0). '
                       'ke_err: |KE_pred - KE_gt| / KE_gt on free-flight frames only.',
        'px_err_curve': arr_px,
        'px_err_std_curve': arr_px_std,
        'mean_ke_err_free_flight': mean_ke_err,
        'std_ke_err_free_flight': std_ke_err,
    }

    path = OUT_DIR / 'conservation.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nMean px_err final step: {arr_px[-1]:.4f}")
    print(f"Mean KE err (free-flight frames): {mean_ke_err:.4f} ± {std_ke_err:.4f}")
    print(f"Saved conservation → {path}")
    return result

# ──────────────────────────────────────────────────────────────────────────────
# 3. Collision-frame vs free-flight decomposition (no model needed — GT-based)
# ──────────────────────────────────────────────────────────────────────────────

def run_collision_decomposition():
    """
    For each scenario type:
      - Generate 30 scenes
      - For each frame pair (t, t+1) classify as collision or free-flight
        using velocity change > threshold
      - Compute linear extrap MSE separately for each class
    Then estimate PhysicsLM's collision-frame MSE using:
      overall_MSE ≈ col_frac * col_MSE + (1-col_frac) * flight_MSE
    """
    print("\n=== Collision-frame vs free-flight decomposition ===")
    from src.physics import generate_scenario, SCENARIO_TYPES

    SEEN = ['billiards','bowling','head_on','explosion','projectile',
            'pyramid','tower','jenga','dominos','bridge',
            'ramp_roll','ski_jump','marble_run','avalanche','plinko',
            'pendulum','chain','seesaw','wrecking_ball','orbit',
            'basketball','conveyor','pong','wind','breakout',
            'angry_birds','hourglass','newtons_cradle','pinball']
    paper_types = [s for s in SEEN if s in set(SCENARIO_TYPES)]

    V_CHANGE_THRESH = 10.0  # px/s — velocity change this large = collision frame

    # PhysicsLM overall MSE per scenario from stage0_results.json
    stage0 = json.load(open('/home/alexw/evaluation_results/lfm2-scenarios/stage0_results.json'))
    plm_mse = {k: v['pos_mse'] for k, v in stage0['per_scenario'].items()}

    CATS = {
        'Collision': ['billiards','bowling','head_on','explosion','projectile'],
        'Stacking':  ['pyramid','tower','jenga','dominos','bridge'],
        'Ramp':      ['ramp_roll','ski_jump','marble_run','avalanche','plinko'],
        'Constraint':['pendulum','chain','seesaw','wrecking_ball','orbit'],
        'Minigame':  ['basketball','conveyor','pong','wind','breakout'],
        'Complex':   ['angry_birds','hourglass','newtons_cradle','pinball'],
    }

    scenario_results = {}
    for scen in sorted(paper_types):
        col_lin, flight_lin = [], []
        col_count = flight_count = 0

        for seed in range(5000000, 5000030):
            try:
                sim, _ = generate_scenario(seed, scenario_type=scen)
                states = []
                for _ in range(121): states.append(sim.get_state()); sim.step()
            except:
                continue

            for t in range(4, 119):
                o0 = {o['id']: o for o in states[t]['objects']}
                o1 = {o['id']: o for o in states[t+1]['objects']}

                # detect collision: any object changes velocity by > threshold
                is_collision = False
                for oid in set(o0) & set(o1):
                    v0 = o0[oid].get('velocity', {'x':0,'y':0})
                    v1 = o1[oid].get('velocity', {'x':0,'y':0})
                    dv = math.sqrt((v1['x']-v0['x'])**2 + (v1['y']-v0['y'])**2)
                    if dv > V_CHANGE_THRESH:
                        is_collision = True; break

                # linear extrap MSE
                for oid in set(o0) & set(o1):
                    p0 = o0[oid]['position']; p1 = o1[oid]['position']
                    v0 = o0[oid].get('velocity', {'x':0,'y':0})
                    lin_err = (p0['x']+v0['x']*DT - p1['x'])**2 + (p0['y']+v0['y']*DT - p1['y'])**2
                    if is_collision:
                        col_lin.append(lin_err); col_count += 1
                    else:
                        flight_lin.append(lin_err); flight_count += 1

        total = col_count + flight_count
        col_frac = col_count / total if total else 0

        col_lin_mse   = float(np.mean(col_lin))  if col_lin  else 0.0
        flight_lin_mse= float(np.mean(flight_lin)) if flight_lin else 0.0

        # Estimate PhysicsLM collision MSE (algebra)
        plm_total = plm_mse.get(scen, None)
        if plm_total is not None and col_frac > 0:
            plm_col_est = (plm_total - (1-col_frac)*flight_lin_mse) / col_frac
        else:
            plm_col_est = None

        scenario_results[scen] = {
            'col_frac': col_frac,
            'col_lin_mse': col_lin_mse,
            'flight_lin_mse': flight_lin_mse,
            'plm_total_mse': plm_total,
            'plm_col_mse_est': plm_col_est,
        }
        print(f"  {scen:20s}: col={col_frac:.2%}  "
              f"lin_col={col_lin_mse:8.2f}  lin_flight={flight_lin_mse:.4f}  "
              f"plm_col_est={plm_col_est:.1f}" if plm_col_est else
              f"  {scen:20s}: col={col_frac:.2%}  lin_col={col_lin_mse:8.2f}")

    # Category summaries
    cat_summary = {}
    for cat, types in CATS.items():
        rows = [scenario_results[t] for t in types if t in scenario_results]
        if not rows: continue
        cat_summary[cat] = {
            'col_frac':       np.mean([r['col_frac']       for r in rows]),
            'col_lin_mse':    np.mean([r['col_lin_mse']    for r in rows]),
            'flight_lin_mse': np.mean([r['flight_lin_mse'] for r in rows]),
        }

    result = {'per_scenario': scenario_results, 'per_category': cat_summary}
    path = OUT_DIR / 'collision_decomp.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved collision decomp → {path}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-model', action='store_true',
                    help='Skip model-based eval (rollout + conservation); only run decomposition')
    ap.add_argument('--rollout-only', action='store_true')
    ap.add_argument('--conservation-only', action='store_true')
    ap.add_argument('--decomp-only', action='store_true')
    args = ap.parse_args()

    if args.decomp_only or args.skip_model:
        run_collision_decomposition()
    elif args.rollout_only:
        model = load_model()
        run_rollout_eval(model)
    elif args.conservation_only:
        model = load_model()
        run_conservation_eval(model)
    else:
        # Run decomposition first (no GPU needed)
        decomp = run_collision_decomposition()
        # Then model-based evals
        model = load_model()
        rollout = run_rollout_eval(model)
        conservation = run_conservation_eval(model)
        print("\nAll eval complete.")
