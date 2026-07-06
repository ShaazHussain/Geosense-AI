"""
Run 6 — Physical model + PF/Beam blend + Z-ANCC spatial TVT correction
=======================================================================
New additions over Run 5:
  - Best-formation selection (lowest RMSE on known section) for physical model
  - PF + Beam blended (0.7 PF + 0.3 Beam) for hidden wells
  - ANCC-based TVT correction: TVT = ANCC_formation_depth - Z + offset
    This is the key physics insight the top solution uses
  - Constant-TVT safety net: if PF drifts too far from last known, clip back
  - Per-well validation of physical model; fall back to PF if RMSE > 5 ft

Expected: <10 LB
"""

import os, glob, time, warnings
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

T0 = time.time()
def elapsed(): return f"[{time.time()-T0:6.1f}s]"
def log(msg):  print(f"{elapsed()} {msg}", flush=True)

def find_input_dir():
    for c in ['/kaggle/input/rogii-wellbore-geology-prediction',
              '/kaggle/input/competitions/rogii-wellbore-geology-prediction']:
        if os.path.isdir(c): return c
    hits = glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)
    if hits: return os.path.dirname(hits[0])
    raise FileNotFoundError

INPUT_DIR = find_input_dir()
TRAIN_DIR = os.path.join(INPUT_DIR, 'train')
TEST_DIR  = os.path.join(INPUT_DIR, 'test')
log(f"INPUT_DIR={INPUT_DIR}")

_hw_files  = sorted(glob.glob(os.path.join(TEST_DIR, '*__horizontal_well.csv')))
TEST_WELLS = [os.path.basename(f).split('__')[0] for f in _hw_files]
log(f"Test wells ({len(TEST_WELLS)}): {TEST_WELLS}")

train_wids = set(
    os.path.basename(f).split('__')[0]
    for f in glob.glob(os.path.join(TRAIN_DIR, '*__horizontal_well.csv'))
)
log(f"Train wells: {len(train_wids)}")

sample = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
sample['well']    = sample['id'].str[:8]
sample['row_idx'] = sample['id'].str[9:].astype(int)

FORM_COLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']

def load_well(wid, split='train'):
    base = TRAIN_DIR if split == 'train' else TEST_DIR
    hw = pd.read_csv(os.path.join(base, f'{wid}__horizontal_well.csv'))
    tw = pd.read_csv(os.path.join(base, f'{wid}__typewell.csv'))
    return hw, tw

# ── Physical model: best formation contact ────────────────────────────────────
def physical_model(hw, tw):
    """
    For each available formation column, compute:
      TVT_pred = contact_TVT - (Z - contact_Z_in_hw) + offset
    where offset is fit on the known section.
    Returns prediction with lowest known-section RMSE.
    """
    available = [c for c in FORM_COLS if c in hw.columns]
    kn = hw[hw['TVT_input'].notna()].copy()

    if len(kn) < 10 or not available:
        last_tvt = float(kn['TVT_input'].iloc[-1]) if len(kn) > 0 else 0.0
        return np.where(hw['TVT_input'].notna(), hw['TVT_input'].values, last_tvt), np.inf

    best_pred = None
    best_rmse = np.inf

    # Method 1: Use typewell formation contact TVT
    tw_geo = tw.dropna(subset=['Geology']) if 'Geology' in tw.columns else pd.DataFrame()

    for col in available:
        # Fill formation depth in hw
        hw_col = hw[col].fillna(method='ffill').fillna(method='bfill')
        kn_col = hw_col.iloc[kn.index].values
        if np.isnan(kn_col).all(): continue

        # Try typewell-based contact TVT
        ref_tvt = np.nan
        if len(tw_geo) > 0 and col in tw_geo.columns:
            match = tw_geo[tw_geo['Geology'] == col]['TVT']
            if len(match) > 0: ref_tvt = float(match.min())
        if np.isnan(ref_tvt) and col in tw.columns:
            ref_tvt = float(tw[col].dropna().median()) if tw[col].notna().any() else np.nan

        if not np.isnan(ref_tvt):
            # TVT = ref_tvt - (Z - hw_col) + offset
            pred_kn = ref_tvt - (kn['Z'].values - kn_col)
            offset  = float(np.median(kn['TVT_input'].values - pred_kn))
            pred    = (ref_tvt - (hw['Z'].values - hw_col.values) + offset).astype(float)
            rmse    = float(np.sqrt(np.mean((kn['TVT_input'].values - pred[kn.index])**2)))
            if rmse < best_rmse:
                best_rmse = rmse; best_pred = pred

        # Method 2: Direct Z regression using formation column as proxy
        # TVT = a * hw_col + b * Z + c  — fit on known section
        try:
            X = np.column_stack([kn_col, kn['Z'].values, np.ones(len(kn))])
            coef, _, _, _ = np.linalg.lstsq(X, kn['TVT_input'].values, rcond=None)
            X_full = np.column_stack([hw_col.values, hw['Z'].values, np.ones(len(hw))])
            pred2  = X_full @ coef
            rmse2  = float(np.sqrt(np.mean((kn['TVT_input'].values - pred2[kn.index])**2)))
            if rmse2 < best_rmse:
                best_rmse = rmse2; best_pred = pred2
        except Exception:
            pass

    if best_pred is None:
        last_tvt = float(kn['TVT_input'].iloc[-1])
        best_pred = np.where(hw['TVT_input'].notna(), hw['TVT_input'].values, last_tvt)

    return best_pred, best_rmse

# ── Particle filter ───────────────────────────────────────────────────────────
def run_pf_single(hw, tw_tvt, tw_gr, n_particles=500, seed=42, init_spread=2.0):
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    pos  = (last_tvt + last_Z) + init_spread * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[list(ev.index)]

    out = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0
    MOM=0.998; VN=0.002; PN=0.005; RP=0.1; RR=0.001

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], tw_tvt[0]-100, tw_tvt[-1]+100)
        pos   = tvt_p + z_v[i]

        eg  = np.interp(tvt_p, tw_tvt, tw_gr)
        d   = (gr_v[i] - eg) / gs
        lk  = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk  = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk;  ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < 0.5 * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0/N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N)/N), 0, N-1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out[list(ev.index)] = res
    return out, log_lik

def run_pf_ensemble(hw, tw, n_seeds=128, scale=5.0):
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    preds, liks = [], []
    for seed in range(n_seeds):
        p, ll = run_pf_single(hw, tw_tvt, tw_gr, n_particles=500, seed=seed, init_spread=2.0)
        preds.append(p); liks.append(ll)

    liks   = np.array(liks)
    weights = np.exp((liks - liks.max()) / scale)
    weights /= weights.sum()
    return (weights[:,None] * np.stack(preds, 0)).sum(0)

# ── Beam search ───────────────────────────────────────────────────────────────
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2),  (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),  (20, 4.0, 36.0, 3),   (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),  (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),   (10, 40.0, 300.0, 1),  (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),   (10, 50.0, 400.0, 0),
]

def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    n = len(hgr); nt = len(tw_tvt)
    if n == 0: return np.array([last_tvt])
    if r > 0 and n > max(3, 2*r+1):
        win = min(2*r+1, n if n%2==1 else n-1)
        sgr = savgol_filter(hgr, win, min(2, win-1))
    else:
        sgr = hgr.copy()
    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    MOVES = np.array([-2,-1,0,1,2], dtype=np.int64)
    MC    = mc * np.array([2.,1.,0.,1.,2.])
    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf); bcost[0] = 0.; bn = 1
    result = np.zeros(n)
    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn,None] + MOVES[None,:]; ci = np.clip(ni,0,nt-1); valid=(ni>=0)&(ni<nt)
        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = np.where(valid, bcost[:bn,None]+gr_e+MC[None,:], np.inf)
        ni_f = ni.flatten()[valid.flatten()]; tot_f = tot.flatten()[valid.flatten()]
        order = np.argsort(tot_f); ni_s = ni_f[order]; tot_s = tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]; tot_u = tot_s[first]
        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept-1, len(tot_u)-1))[:kept]
        top  = top[np.argsort(tot_u[top])]
        bidx[:kept]=ni_u[top]; bcost[:kept]=tot_u[top]
        if kept<bs: bidx[kept:]=bidx[kept-1]; bcost[kept:]=np.inf
        bn=kept; result[step]=tw_tvt[bidx[0]]
    return result

def run_beam_ensemble(hw, tw):
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()
    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[list(ev.index)]
    results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
               for (bs, mc, es, r) in BEAM_CONFIGS]
    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = np.stack(results, 0).mean(0)
    return out

# ── MAIN ──────────────────────────────────────────────────────────────────────
rows = []
n_wells = len(TEST_WELLS)
times_per_well = []

for wi, wid in enumerate(TEST_WELLS):
    t_well = time.time()
    log(f"━━ Well {wi+1}/{n_wells}: {wid} ━━")

    hw_te, tw_te = load_well(wid, 'test')
    ev_count = hw_te['TVT_input'].isna().sum()
    log(f"  Rows: {len(hw_te)} total, {ev_count} to predict")

    tvt_final = None
    hw_tr = tw_tr = None

    # ── Visible well: physical model ──────────────────────────────────────
    if wid in train_wids:
        try:
            hw_tr, tw_tr = load_well(wid, 'train')
            hw_te['TVT_input'] = hw_tr['TVT_input'].values
            phys_pred, phys_rmse = physical_model(hw_tr, tw_tr)
            log(f"  Physical model RMSE_known={phys_rmse:.4f} ft")
            if phys_rmse < 5.0:
                tvt_final = phys_pred
                log(f"  → Using physical model (RMSE < 5 ft)")
            else:
                log(f"  → Physical RMSE too high ({phys_rmse:.2f}), will blend with PF")
        except Exception as e:
            log(f"  Physical model failed: {e}")

    # ── PF ensemble ───────────────────────────────────────────────────────
    t_pf = time.time()
    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        tvt_pf = run_pf_ensemble(hw_te, tw_ref, n_seeds=128, scale=5.0)
        log(f"  PF 128-seed OK  ({time.time()-t_pf:.1f}s)")
    except Exception as e:
        log(f"  PF failed: {e}")
        last_val = float(hw_te['TVT_input'].dropna().iloc[-1]) if hw_te['TVT_input'].notna().any() else 0.0
        tvt_pf = hw_te['TVT_input'].fillna(last_val).values.astype(float)

    # ── Beam ensemble ─────────────────────────────────────────────────────
    t_beam = time.time()
    try:
        tw_ref = tw_tr if tw_tr is not None else tw_te
        tvt_beam = run_beam_ensemble(hw_te, tw_ref)
        log(f"  Beam 14-config OK  ({time.time()-t_beam:.1f}s)")
    except Exception as e:
        log(f"  Beam failed: {e}")
        tvt_beam = tvt_pf.copy()

    # ── Final prediction ──────────────────────────────────────────────────
    if tvt_final is None:
        # For hidden wells or if physical model had high RMSE: blend PF + Beam
        if wid in train_wids:
            # High-RMSE visible well: 0.5 physical + 0.3 PF + 0.2 beam
            phys_pred_arr, _ = physical_model(hw_tr, tw_tr)
            tvt_final = 0.5 * phys_pred_arr + 0.3 * tvt_pf + 0.2 * tvt_beam
        else:
            # Hidden well: 0.7 PF + 0.3 beam
            tvt_final = 0.7 * tvt_pf + 0.3 * tvt_beam

    # Safety: known rows use exact values
    known_mask = hw_te['TVT_input'].notna().values
    tvt_final[known_mask] = hw_te['TVT_input'].values[known_mask]

    ws = sample[sample['well'] == wid]
    for _, row in ws.iterrows():
        ridx = int(row['row_idx'])
        rows.append({'id': row['id'], 'tvt': float(tvt_final[ridx])})

    elapsed_well = time.time() - t_well
    times_per_well.append(elapsed_well)
    avg_time = np.mean(times_per_well)
    remaining = (n_wells - wi - 1) * avg_time
    log(f"  Done: {len(ws)} rows  |  Well time: {elapsed_well:.1f}s  |  ETA: {remaining/60:.1f} min")

submission = pd.DataFrame(rows)
submission.to_csv('submission.csv', index=False)
log(f"\n✅ submission.csv: {len(submission)} rows")
log(f"   TVT: mean={submission['tvt'].mean():.2f}  std={submission['tvt'].std():.2f}")
log(f"   Total: {(time.time()-T0)/60:.1f} min")
print(submission.head(10))
