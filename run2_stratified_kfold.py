# ═══════════════════════════════════════════════════════════
# RUN 2 — Stratified K-Fold by TVT band + residual target
# Key idea: wells cluster into TVT bands (seen in EDA).
# Stratify folds by band so each fold sees all depth ranges.
# OOF predictions used as the final val score.
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

SEED   = 42
N_FOLD = 5

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

def make_feats(h, tw):
    df = h.copy()
    ps = ps_idx(df)
    last_tvt = df.loc[ps, 'TVT_input']
    known    = df.loc[:ps, ['MD','TVT_input']].dropna()

    dips = {}
    for tail_n, key in [(20,'dip20'), (50,'dip50'), (100,'dip100')]:
        tail = known.tail(tail_n)
        dips[key] = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail)>2 else 0.0
    dip_mean = np.mean(list(dips.values()))

    df['dmd_ps'] = df['MD'] - df.loc[ps, 'MD']
    for key, d in dips.items():
        df[f'tvt_extrap_{key}'] = last_tvt + d * df['dmd_ps']
    df['tvt_extrap_mean'] = last_tvt + dip_mean * df['dmd_ps']
    df['dip20'] = dips['dip20']; df['dip50'] = dips['dip50']; df['dip100'] = dips['dip100']
    df['dip_spread'] = max(dips.values()) - min(dips.values())

    gr = df['GR'].interpolate().bfill().ffill()
    df['gr'] = gr
    for w in [5, 10, 20, 40, 80]:
        df[f'gr_rm{w}'] = gr.rolling(w, min_periods=1).mean()
        df[f'gr_rs{w}'] = gr.rolling(w, min_periods=1).std().fillna(0)
    df['gr_d1'] = gr.diff().fillna(0)
    df['gr_d2'] = df['gr_d1'].diff().fillna(0)

    gr_mu  = gr.iloc[:ps+1].tail(50).mean()
    gr_sig = gr.iloc[:ps+1].tail(50).std() + 1e-6
    df['gr_zscore'] = (gr - gr_mu) / gr_sig

    df['tvt_filled']    = df['TVT_input'].interpolate().bfill().ffill()
    df['tvt_last']      = last_tvt
    df['steps_from_ps'] = (np.arange(len(df)) - ps).clip(min=0)
    df['dMD'] = df['MD'].diff().fillna(0)
    df['tw_gr']       = tw_interp(tw, df['tvt_filled'].values)
    df['tw_residual'] = df['gr'] - df['tw_gr']

    for c in ['X','Y','Z']:
        if c in df.columns:
            df[f'd{c}'] = df[c].diff().fillna(0)

    base = ['MD','dMD','dmd_ps','steps_from_ps','gr','gr_d1','gr_d2','gr_zscore',
            'gr_rm5','gr_rs5','gr_rm10','gr_rs10','gr_rm20','gr_rs20',
            'gr_rm40','gr_rs40','gr_rm80','gr_rs80',
            'tvt_filled','tvt_last','tw_gr','tw_residual',
            'dip20','dip50','dip100','dip_spread',
            'tvt_extrap_dip20','tvt_extrap_dip50','tvt_extrap_dip100','tvt_extrap_mean']
    xyz = [c for c in ['X','Y','Z','dX','dY','dZ'] if c in df.columns]

    if 'TVT' in df.columns:
        df['tvt_resid'] = df['TVT'] - df['tvt_extrap_mean']

    return df, base + xyz

rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))

# ── stratify wells by median TVT band ─────────────────────────
print('Loading wells …')
all_w = load_wells(TRAIN)
if not all_w:
    print(f"ERROR: No wells found in {TRAIN}")
    print(f"  Looking for files matching: *__horizontal_well.csv")
    print(f"  Please ensure TRAIN path is correct and data files exist.")
    raise ValueError(f"No wells loaded from {TRAIN}")

# compute median TVT per well for stratification
well_meta = []
for wid, h, tw in all_w:
    med_tvt = h['TVT'].median()
    well_meta.append({'wid': wid, 'med_tvt': med_tvt})
meta_df = pd.DataFrame(well_meta)

# bin into N_FOLD quantile bands
meta_df['band'] = pd.qcut(meta_df['med_tvt'], q=N_FOLD, labels=False)
print('Wells per band:', meta_df['band'].value_counts().sort_index().to_dict())

# assign fold: round-robin within each band
meta_df['fold'] = -1
for band, grp in meta_df.groupby('band'):
    shuffled = grp.sample(frac=1, random_state=SEED)
    for i, idx in enumerate(shuffled.index):
        meta_df.loc[idx, 'fold'] = i % N_FOLD

# create fold lookup
fold_map = dict(zip(meta_df['wid'], meta_df['fold']))
print('Wells per fold:', meta_df['fold'].value_counts().sort_index().to_dict())

# build full well lookup for quick access
well_lookup = {wid: (h, tw) for wid, h, tw in all_w}

# ── K-Fold training ───────────────────────────────────────────
oof_true, oof_pred_resid, oof_extrap = [], [], []
models, meds, fcolss = [], [], []

lgb_params = dict(
    n_estimators=1200,
    learning_rate=0.025,
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

for fold in range(N_FOLD):
    tr_ids  = meta_df[meta_df['fold'] != fold]['wid'].tolist()
    val_ids = meta_df[meta_df['fold'] == fold]['wid'].tolist()

    def make_ds(ids):
        Xs, ys = [], []
        fc = None
        for wid in ids:
            h, tw = well_lookup[wid]
            ps = ps_idx(h)
            df, fc_ = make_feats(h, tw)
            fc = fc_
            zone = df.iloc[ps+1:]
            if len(zone)==0: continue
            Xs.append(zone[fc_].values)
            ys.append(zone['tvt_resid'].values)
        X = np.vstack(Xs); y = np.concatenate(ys)
        med_ = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]): X[np.isnan(X[:,j]),j]=med_[j]
        return X, y, fc, med_

    X_tr, y_tr, fcols, med_ = make_ds(tr_ids)
    X_vl, y_vl, _,     _    = make_ds(val_ids)
    for j in range(X_vl.shape[1]): X_vl[np.isnan(X_vl[:,j]),j]=med_[j]

    m = LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_vl, y_vl)],
          callbacks=[early_stopping(60, verbose=False), log_evaluation(200)])

    # collect OOF predictions
    for wid in val_ids:
        h, tw = well_lookup[wid]
        ps = ps_idx(h)
        df, fc = make_feats(h, tw)
        zone = df.iloc[ps+1:]
        if len(zone)==0: continue
        X_w = zone[fc].values
        for j in range(X_w.shape[1]): X_w[np.isnan(X_w[:,j]),j]=med_[j]
        r_pred = m.predict(X_w)
        oof_pred_resid.extend(r_pred)
        oof_extrap.extend(zone['tvt_extrap_mean'].values)
        oof_true.extend(zone['TVT'].values)

    fold_tvt_true = np.array(oof_true[-len(val_ids):] if False else [])  # per-fold print
    fold_rmse = rmse(
        np.array(oof_true[-sum(len(df.iloc[ps_idx(well_lookup[w][0])+1:]) for w in val_ids):]),
        np.array(oof_extrap[-sum(len(df.iloc[ps_idx(well_lookup[w][0])+1:]) for w in val_ids):]) +
        np.array(oof_pred_resid[-sum(len(df.iloc[ps_idx(well_lookup[w][0])+1:]) for w in val_ids):])
    ) if False else 0.0

    print(f'  Fold {fold}: val wells={len(val_ids)} | best iter={m.best_iteration_}')
    models.append(m); meds.append(med_); fcolss.append(fcols)

oof_tvt_pred = np.array(oof_extrap) + np.array(oof_pred_resid)
print(f'\nOOF TVT RMSE (stratified KFold, N={N_FOLD}): {rmse(np.array(oof_true), oof_tvt_pred):.4f} ft')

# ── inference: average across folds ──────────────────────────
sample_sub = pd.read_csv(SUB)
sample_sub[['well_hash','md_int']] = sample_sub['id'].str.rsplit('_', n=1, expand=True)
sample_sub['md_int'] = sample_sub['md_int'].astype(int)
sample_sub['tvt'] = 0.0

for wid, h, tw in load_wells(TEST):
    ps = ps_idx(h)
    df, fc = make_feats(h, tw)
    zone = df.iloc[ps+1:].copy()
    if zone.empty: continue

    fold_preds = []
    for m, med_, fc_ in zip(models, meds, fcolss):
        X_t = zone[fc_].values.copy()
        for j in range(X_t.shape[1]): X_t[np.isnan(X_t[:,j]),j]=med_[j]
        fold_preds.append(m.predict(X_t))
    resid_avg = np.mean(fold_preds, axis=0)
    pred = zone['tvt_extrap_mean'].values + resid_avg

    wh = wid[:8]
    sub_idx = sample_sub[sample_sub['well_hash']==wh].sort_values('md_int').index
    min_len = min(len(sub_idx), len(pred))
    sample_sub.loc[sub_idx[:min_len], 'tvt'] = pred[:min_len]

sample_sub[['id','tvt']].to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f'\nRun 2 done. Submission saved to {OUTPUT_DIR / "submission.csv"}')
print(f'  {len(sample_sub)} rows')
print(sample_sub['tvt'].describe())
