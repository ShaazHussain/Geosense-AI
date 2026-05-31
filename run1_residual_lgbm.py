# ═══════════════════════════════════════════════════════════
# RUN 1 — Residual LightGBM
# Key idea: predict (TVT - linear_extrap) instead of raw TVT.
# Since ΔTVT ≈ 0, the residual is tiny → much easier target.
# ═══════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings, sys
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS  (auto-detect Kaggle vs local)
# ─────────────────────────────────────────────
if os.path.exists('/kaggle/input'):
    # On Kaggle, search for the train directory dynamically
    kaggle_input = Path('/kaggle/input')
    train_dirs = list(kaggle_input.rglob('train'))
    
    if train_dirs:
        # Found a train directory, use its parent as BASE
        BASE = train_dirs[0].parent
    else:
        # Fallback to original expected path
        BASE = Path('/kaggle/input/rogii-wellbore-geology-prediction')
    
    TRAIN = BASE / 'train'
    TEST  = BASE / 'test'
    SUB   = BASE / 'sample_submission.csv'
    OUTPUT_DIR = Path('/kaggle/working')
else:
    # Try to find the data directory - look for 'train' folder in current dir or parent
    BASE = Path.cwd()
    if not (BASE / 'train').exists():
        # Check if we're in a subdirectory, look for train in parent
        if (BASE.parent / 'train').exists():
            BASE = BASE.parent
    
    TRAIN = BASE / 'train'
    TEST  = BASE / 'test'
    SUB   = BASE / 'sample_submission.csv'
    OUTPUT_DIR = BASE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"BASE: {BASE}")
print(f"TRAIN exists: {TRAIN.exists()}")
print(f"TEST exists: {TEST.exists()}")
print(f"SUB exists: {SUB.exists()}")

SEED, VAL_FRAC = 42, 0.2

# ── loaders (unchanged) ──────────────────────────────────────
def load_wells(d):
    out = []
    for hf in sorted(Path(d).glob('*__horizontal_well.csv')):
        wid = hf.stem.replace('__horizontal_well','')
        tws = list(Path(d).glob(f'{wid}__typewell.csv'))
        if not tws: continue
        out.append((wid, pd.read_csv(hf), pd.read_csv(tws[0])))
    return out

def ps_idx(h):
    v = h['TVT_input'].notna()
    return int(v[v].index[-1]) if v.any() else 0

def tw_interp(tw, tvt_q):
    s = tw.sort_values('TVT').dropna(subset=['TVT','GR'])
    return np.interp(tvt_q, s['TVT'].values, s['GR'].values)

# ── NEW feature engineering ───────────────────────────────────
def make_feats(h, tw):
    df = h.copy()
    ps = ps_idx(df)
    last_tvt = df.loc[ps, 'TVT_input']

    # ── dip estimation: fit last 20, 50, 100 pts → ensemble ──
    known = df.loc[:ps, ['MD','TVT_input']].dropna()
    dips = {}
    for tail_n, key in [(20,'dip20'), (50,'dip50'), (100,'dip100')]:
        tail = known.tail(tail_n)
        if len(tail) > 2:
            dips[key] = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0]
        else:
            dips[key] = 0.0
    dip_mean = np.mean(list(dips.values()))

    # MD offset from PS
    df['dmd_ps'] = df['MD'] - df.loc[ps, 'MD']

    # Linear extrapolations using each dip
    for key, d in dips.items():
        df[f'tvt_extrap_{key}'] = last_tvt + d * df['dmd_ps']
    df['tvt_extrap_mean'] = last_tvt + dip_mean * df['dmd_ps']

    df['dip20']  = dips['dip20']
    df['dip50']  = dips['dip50']
    df['dip100'] = dips['dip100']
    df['dip_spread'] = max(dips.values()) - min(dips.values())  # dip stability

    # ── GR features (multi-scale) ─────────────────────────────
    gr = df['GR'].interpolate().bfill().ffill()
    df['gr'] = gr
    for w in [5, 10, 20, 40, 80]:
        df[f'gr_rm{w}']  = gr.rolling(w, min_periods=1).mean()
        df[f'gr_rs{w}']  = gr.rolling(w, min_periods=1).std().fillna(0)
    df['gr_d1'] = gr.diff().fillna(0)
    df['gr_d2'] = df['gr_d1'].diff().fillna(0)

    # GR z-score relative to last 50 pts before PS (within-formation normalisation)
    gr_before_ps = gr.iloc[:ps+1]
    gr_mu  = gr_before_ps.tail(50).mean()
    gr_sig = gr_before_ps.tail(50).std() + 1e-6
    df['gr_zscore'] = (gr - gr_mu) / gr_sig

    # ── Typewell anchor ───────────────────────────────────────
    df['tvt_filled'] = df['TVT_input'].interpolate().bfill().ffill()
    df['tvt_last']   = last_tvt
    df['tw_gr']      = tw_interp(tw, df['tvt_filled'].values)
    df['tw_residual']= df['gr'] - df['tw_gr']

    # ── Positional ────────────────────────────────────────────
    df['dMD'] = df['MD'].diff().fillna(0)
    df['steps_from_ps'] = (np.arange(len(df)) - ps).clip(min=0)
    for c in ['X','Y','Z']:
        if c in df.columns:
            df[f'd{c}'] = df[c].diff().fillna(0)

    base = (['MD','dMD','dmd_ps','steps_from_ps','gr','gr_d1','gr_d2','gr_zscore',
             'gr_rm5','gr_rs5','gr_rm10','gr_rs10','gr_rm20','gr_rs20',
             'gr_rm40','gr_rs40','gr_rm80','gr_rs80',
             'tvt_filled','tvt_last',
             'tw_gr','tw_residual',
             'dip20','dip50','dip100','dip_spread',
             'tvt_extrap_dip20','tvt_extrap_dip50','tvt_extrap_dip100','tvt_extrap_mean'])
    xyz = [c for c in ['X','Y','Z','dX','dY','dZ'] if c in df.columns]
    feats = base + xyz

    # ── TARGET: residual from mean extrapolation ──────────────
    # Model predicts (TVT - tvt_extrap_mean); add back at inference
    if 'TVT' in df.columns:
        df['tvt_resid'] = df['TVT'] - df['tvt_extrap_mean']

    return df, feats

# ── dataset builder ───────────────────────────────────────────
def build_dataset(wells, target_col='tvt_resid'):
    Xs, ys, groups = [], [], []
    fcols = None
    for wid, h, tw in wells:
        ps = ps_idx(h)
        df, fc = make_feats(h, tw)
        fcols = fc
        zone  = df.iloc[ps+1:]
        if len(zone) == 0: continue
        Xs.append(zone[fc].values)
        ys.append(zone[target_col].values)
        groups += [wid]*len(zone)
    X = np.vstack(Xs); y = np.concatenate(ys)
    med = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:,j]), j] = med[j]
    return X, y, groups, fcols, med

rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))

# ── split ─────────────────────────────────────────────────────
print('Loading wells …')
all_w = load_wells(TRAIN)
if not all_w:
    print(f"ERROR: No wells found in {TRAIN}")
    print(f"  Looking for files matching: *__horizontal_well.csv")
    print(f"  Please ensure TRAIN path is correct and data files exist.")
    raise ValueError(f"No wells loaded from {TRAIN}")
np.random.seed(SEED); np.random.shuffle(all_w)
n_val = max(1, int(len(all_w)*VAL_FRAC))
val_w, tr_w = all_w[:n_val], all_w[n_val:]
print(f'  Train: {len(tr_w)}  Val: {len(val_w)}')

X_tr, y_tr, _, fcols, med = build_dataset(tr_w)
X_vl, y_vl, _, _,     _   = build_dataset(val_w)
for j in range(X_vl.shape[1]): X_vl[np.isnan(X_vl[:,j]),j] = med[j]
print(f'  X_train: {X_tr.shape}   X_val: {X_vl.shape}')
print(f'  Residual target  mean={y_tr.mean():.4f}  std={y_tr.std():.4f}  max_abs={np.abs(y_tr).max():.4f}')

# ── model ─────────────────────────────────────────────────────
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1
)
print('\nTraining LightGBM (residual target) …')
model.fit(X_tr, y_tr,
          eval_set=[(X_vl, y_vl)],
          callbacks=[__import__('lightgbm').early_stopping(50, verbose=False),
                     __import__('lightgbm').log_evaluation(100)])

# reconstruct raw TVT for scoring
def predict_tvt(model, wells, med, fcols):
    all_true, all_pred = [], []
    for wid, h, tw in wells:
        ps = ps_idx(h)
        df, fc = make_feats(h, tw)
        zone = df.iloc[ps+1:]
        if len(zone) == 0: continue
        X_w = zone[fc].values
        for j in range(X_w.shape[1]): X_w[np.isnan(X_w[:,j]),j] = med[j]
        resid_pred = model.predict(X_w)
        tvt_pred   = zone['tvt_extrap_mean'].values + resid_pred
        all_pred.extend(tvt_pred)
        if 'TVT' in zone.columns:
            all_true.extend(zone['TVT'].values)
    return np.array(all_true), np.array(all_pred)

tr_true, tr_pred = predict_tvt(model, tr_w, med, fcols)
vl_true, vl_pred = predict_tvt(model, val_w, med, fcols)
print(f'\n  Train TVT RMSE : {rmse(tr_true, tr_pred):.4f} ft')
print(f'  Val   TVT RMSE : {rmse(vl_true, vl_pred):.4f} ft')

# ── feature importance ────────────────────────────────────────
fi = pd.Series(model.feature_importances_, index=fcols).sort_values(ascending=False)
print('\nTop 20 features:')
print(fi.head(20).round(4).to_string())

# ── submission ────────────────────────────────────────────────
sample_sub = pd.read_csv(SUB)
sample_sub[['well_hash','md_int']] = sample_sub['id'].str.rsplit('_', n=1, expand=True)
sample_sub['md_int'] = sample_sub['md_int'].astype(int)
sample_sub['tvt'] = 0.0

for wid, h, tw in load_wells(TEST):
    ps = ps_idx(h)
    df, fc = make_feats(h, tw)
    zone = df.iloc[ps+1:].copy()
    if zone.empty: continue
    X_t = zone[fc].values.copy()
    for j in range(X_t.shape[1]): X_t[np.isnan(X_t[:,j]),j]=med[j]
    pred = zone['tvt_extrap_mean'].values + model.predict(X_t)
    wh = wid[:8]
    sub_idx = sample_sub[sample_sub['well_hash']==wh].sort_values('md_int').index
    min_len = min(len(sub_idx), len(pred))
    sample_sub.loc[sub_idx[:min_len], 'tvt'] = pred[:min_len]

sub_path = OUTPUT_DIR / 'submission.csv'
sample_sub[['id','tvt']].to_csv(sub_path, index=False)
print(f'\nRun 1 done. Submission saved to {sub_path}')
print(f'  {len(sample_sub)} rows')

# ── val plots (skip on headless systems like Kaggle) ────────────────────────────────
try:
    fig, axes = plt.subplots(min(3,len(val_w)), 1, figsize=(14, 5*min(3,len(val_w))))
    if min(3,len(val_w))==1: axes=[axes]
    for ax, (wid, h, tw) in zip(axes, val_w[:3]):
        ps = ps_idx(h)
        df, fc = make_feats(h, tw)
        zone = df.iloc[ps+1:]
        X_w = zone[fc].values
        for j in range(X_w.shape[1]): X_w[np.isnan(X_w[:,j]),j]=med[j]
        resid = model.predict(X_w)
        pred  = zone['tvt_extrap_mean'].values + resid
        r     = rmse(zone['TVT'].values, pred)
        ax.plot(h.loc[:ps,'MD'], h.loc[:ps,'TVT_input'], color='#e9c46a', lw=1.5, label='known')
        ax.plot(zone['MD'], zone['TVT'], color='#0077b6', lw=1.5, label='truth')
        ax.plot(zone['MD'], pred,        color='#e76f51', lw=1.5, ls='--', label=f'pred (RMSE={r:.2f})')
        ax.axvline(h.loc[ps,'MD'], color='red', ls=':', lw=1.2)
        ax.set_title(wid); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'run1_val_plots.png', dpi=100, bbox_inches='tight')
    print('  Validation plots saved.')
except Exception as e:
    print(f'  Skipped plotting: {e}')

print(f'\nRun 1 complete.')
print(sample_sub['tvt'].describe())
