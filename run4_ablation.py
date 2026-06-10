# ═══════════════════════════════════════════════════════════
# RUN 4 — Feature Ablation: What does the ML actually add?
# Compares 4 variants side-by-side on the SAME val split:
#   A. Linear dip only (no ML)
#   B. Constant TVT only (no ML)
#   C. ML on {dip + positional} only (no GR)
#   D. ML on {dip + GR} only (no typewell)
# Submits the best variant.
# This tells us exactly which feature groups matter.
# ═══════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
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

SEED  = 42

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

def compute_base(h, tw):
    """Compute all columns; caller selects feature subset."""
    df = h.copy()
    ps = ps_idx(df)
    last_tvt = df.loc[ps, 'TVT_input']
    known    = df.loc[:ps, ['MD','TVT_input']].dropna()

    dips = {}
    for tail_n, key in [(20,'dip20'),(50,'dip50'),(100,'dip100')]:
        t = known.tail(tail_n)
        dips[key] = np.polyfit(t['MD'], t['TVT_input'], 1)[0] if len(t)>2 else 0.
    dip_mean = np.mean(list(dips.values()))

    df['dmd_ps']          = df['MD'] - df.loc[ps,'MD']
    df['tvt_extrap_mean'] = last_tvt + dip_mean * df['dmd_ps']
    df['dip_mean']        = dip_mean
    df['dip20'] = dips['dip20']; df['dip50'] = dips['dip50']; df['dip100'] = dips['dip100']
    df['dip_spread'] = max(dips.values()) - min(dips.values())
    df['steps_from_ps'] = (np.arange(len(df)) - ps).clip(min=0)
    df['dMD'] = df['MD'].diff().fillna(0)
    df['tvt_last'] = last_tvt

    gr = df['GR'].interpolate().bfill().ffill()
    df['gr'] = gr
    for w in [5,10,20,40,80]:
        df[f'gr_rm{w}'] = gr.rolling(w,min_periods=1).mean()
        df[f'gr_rs{w}'] = gr.rolling(w,min_periods=1).std().fillna(0)
    df['gr_d1'] = gr.diff().fillna(0)
    df['gr_d2'] = df['gr_d1'].diff().fillna(0)
    gr_mu  = gr.iloc[:ps+1].tail(50).mean()
    gr_sig = gr.iloc[:ps+1].tail(50).std() + 1e-6
    df['gr_zscore'] = (gr - gr_mu) / gr_sig

    df['tvt_filled'] = df['TVT_input'].interpolate().bfill().ffill()
    df['tw_gr']      = tw_interp(tw, df['tvt_filled'].values)
    df['tw_residual']= df['gr'] - df['tw_gr']

    for c in ['X','Y','Z']:
        if c in df.columns:
            df[f'd{c}'] = df[c].diff().fillna(0)

    if 'TVT' in df.columns:
        df['tvt_resid'] = df['TVT'] - df['tvt_extrap_mean']

    return df, ps

# Feature subsets for ablation
FEAT_SETS = {
    'C_dip_pos_only': ['MD','dMD','dmd_ps','steps_from_ps','tvt_last',
                        'dip20','dip50','dip100','dip_spread','tvt_extrap_mean'],
    'D_dip_gr_notw':  ['MD','dMD','dmd_ps','steps_from_ps','tvt_last',
                        'dip20','dip50','dip100','dip_spread','tvt_extrap_mean',
                        'gr','gr_d1','gr_d2','gr_zscore',
                        'gr_rm5','gr_rs5','gr_rm10','gr_rs10','gr_rm20','gr_rs20',
                        'gr_rm40','gr_rs40','gr_rm80','gr_rs80'],
    'E_full':         ['MD','dMD','dmd_ps','steps_from_ps','tvt_last',
                        'dip20','dip50','dip100','dip_spread','tvt_extrap_mean',
                        'gr','gr_d1','gr_d2','gr_zscore',
                        'gr_rm5','gr_rs5','gr_rm10','gr_rs10','gr_rm20','gr_rs20',
                        'gr_rm40','gr_rs40','gr_rm80','gr_rs80',
                        'tw_gr','tw_residual'],
}
# add XYZ if present (checked at build time)

rmse = lambda a, b: np.sqrt(mean_squared_error(a,b))

# ── fixed val split ────────────────────────────────────────────
print('Loading wells …')
all_w = load_wells(TRAIN)
if not all_w:
    print(f"ERROR: No wells found in {TRAIN}")
    print(f"  Looking for files matching: *__horizontal_well.csv")
    print(f"  Please ensure TRAIN path is correct and data files exist.")
    raise ValueError(f"No wells loaded from {TRAIN}")
np.random.seed(SEED); np.random.shuffle(all_w)
n_val = max(1, int(len(all_w)*0.2))
val_w, tr_w = all_w[:n_val], all_w[n_val:]
print(f'Train: {len(tr_w)}  Val: {len(val_w)}')

# ── Rule-based baselines (A, B) ───────────────────────────────
rule_results = {}
for rule_name, use_dip in [('A_linear_dip', True), ('B_constant', False)]:
    trues, preds = [], []
    for wid, h, tw in val_w:
        df, ps = compute_base(h, tw)
        zone = df.iloc[ps+1:]
        if len(zone)==0: continue
        if use_dip:
            p = zone['tvt_extrap_mean'].values
        else:
            p = np.full(len(zone), df.loc[ps,'TVT_input'])
        trues.extend(zone['TVT'].values)
        preds.extend(p)
    r = rmse(np.array(trues), np.array(preds))
    rule_results[rule_name] = r
    print(f'  {rule_name}: Val RMSE = {r:.4f} ft')

# ── ML variants (C, D, E) ─────────────────────────────────────
lgb_params = dict(n_estimators=1000, learning_rate=0.03, max_depth=6,
                  num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                  min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=SEED, verbose=-1, n_jobs=-1)

ml_results = {}
best_model_name = None
best_model_rmse = 999
best_model_obj  = None
best_med        = None
best_fcols      = None

for variant, feat_cols in FEAT_SETS.items():
    def build_ds(wells, fc):
        Xs, ys = [], []
        for wid, h, tw in wells:
            df, ps = compute_base(h, tw)
            # add XYZ if available
            extra = [c for c in ['X','Y','Z','dX','dY','dZ'] if c in df.columns and c not in fc]
            fc_use = fc + extra
            zone = df.iloc[ps+1:]
            if len(zone)==0: continue
            Xs.append(zone[fc_use].values)
            ys.append(zone['tvt_resid'].values)
        X = np.vstack(Xs); y = np.concatenate(ys)
        med_ = np.nanmedian(X,axis=0)
        for j in range(X.shape[1]): X[np.isnan(X[:,j]),j]=med_[j]
        return X, y, fc_use, med_

    X_tr, y_tr, fc_use, med_ = build_ds(tr_w, feat_cols[:])
    X_vl, y_vl, _,      _    = build_ds(val_w, feat_cols[:])
    for j in range(X_vl.shape[1]): X_vl[np.isnan(X_vl[:,j]),j]=med_[j]

    m = LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_vl,y_vl)],
          callbacks=[early_stopping(50,verbose=False), log_evaluation(300)])

    # reconstruct TVT for scoring
    trues, preds = [], []
    for wid, h, tw in val_w:
        df, ps = compute_base(h, tw)
        extra = [c for c in ['X','Y','Z','dX','dY','dZ'] if c in df.columns and c not in feat_cols]
        fc_use2 = feat_cols + extra
        zone = df.iloc[ps+1:]
        if len(zone)==0: continue
        X_w = zone[fc_use2].values.copy()
        for j in range(X_w.shape[1]): X_w[np.isnan(X_w[:,j]),j]=med_[j]
        pred = zone['tvt_extrap_mean'].values + m.predict(X_w)
        trues.extend(zone['TVT'].values)
        preds.extend(pred)
    r = rmse(np.array(trues), np.array(preds))
    ml_results[variant] = r
    print(f'  {variant}: Val RMSE = {r:.4f} ft  (best_iter={m.best_iteration_})')

    if r < best_model_rmse:
        best_model_rmse  = r
        best_model_name  = variant
        best_model_obj   = m
        best_med         = med_
        best_fcols       = fc_use2

# ── Summary table ─────────────────────────────────────────────
all_results = {**rule_results, **ml_results}
summary = pd.DataFrame({'variant': list(all_results.keys()),
                         'val_rmse': list(all_results.values())}).sort_values('val_rmse')
print('\n' + '='*45)
print('ABLATION SUMMARY (Val RMSE, ft):')
print(summary.to_string(index=False))
print('='*45)

fig, ax = plt.subplots(figsize=(8,4))
ax.barh(summary['variant'], summary['val_rmse'], color='#0077b6')
ax.invert_yaxis()
ax.set_xlabel('Val RMSE (ft)')
ax.set_title('Feature Ablation — Val RMSE')
plt.tight_layout(); plt.show()

# ── Submit best ML variant ─────────────────────────────────────
print(f'\nBest variant: {best_model_name} (RMSE={best_model_rmse:.4f})')

sample_sub = pd.read_csv(SUB)
sample_sub[['well_hash','md_int']] = sample_sub['id'].str.rsplit('_',n=1,expand=True)
sample_sub['md_int'] = sample_sub['md_int'].astype(int)
sample_sub['tvt'] = 0.0

for wid, h, tw in load_wells(TEST):
    df, ps = compute_base(h, tw)
    zone = df.iloc[ps+1:].copy()
    if zone.empty: continue
    X_t = zone[best_fcols].values.copy()
    for j in range(X_t.shape[1]): X_t[np.isnan(X_t[:,j]),j]=best_med[j]
    pred = zone['tvt_extrap_mean'].values + best_model_obj.predict(X_t)
    wh = wid[:8]
    sub_idx = sample_sub[sample_sub['well_hash']==wh].sort_values('md_int').index
    min_len = min(len(sub_idx),len(pred))
    sample_sub.loc[sub_idx[:min_len],'tvt'] = pred[:min_len]

sample_sub[['id','tvt']].to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f'Run 4 done. Submission saved to {OUTPUT_DIR / "submission.csv"}')
print(f'  {len(sample_sub)} rows')
print(sample_sub['tvt'].describe())
