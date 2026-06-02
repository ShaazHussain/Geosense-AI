# ═══════════════════════════════════════════════════════════
# RUN 3 — Typewell cross-correlation TVT correction feature
# Key idea: GR in the horizontal well should match the typewell
# GR at the correct TVT. We cross-correlate a local GR window
# against the typewell to find the best TVT match → this gives
# a physics-informed TVT correction that the model refines.
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

# ── Cross-correlation: find best TVT match in typewell ────────
def xcorr_tvt_estimate(gr_window, tw_gr_arr, tw_tvt_arr, center_tvt, search_range=30):
    """
    Slide a GR window from the horizontal well against typewell GR.
    Returns the typewell TVT where correlation is maximised.
    """
    half = len(gr_window) // 2
    # restrict search to ±search_range ft from current estimate
    mask = np.abs(tw_tvt_arr - center_tvt) < search_range
    if mask.sum() < len(gr_window) + 2:
        return center_tvt, 0.0
    tw_gr_local = tw_gr_arr[mask]
    tw_tvt_local = tw_tvt_arr[mask]
    if len(tw_gr_local) < len(gr_window):
        return center_tvt, 0.0
    best_corr, best_tvt = -np.inf, center_tvt
    # normalise GR window
    gw = gr_window - gr_window.mean()
    gw_std = gw.std() + 1e-6
    for i in range(len(tw_gr_local) - len(gr_window)):
        tw_seg = tw_gr_local[i:i+len(gr_window)]
        tw_seg = tw_seg - tw_seg.mean()
        corr   = np.dot(gw, tw_seg) / (gw_std * (tw_seg.std()+1e-6) * len(gw))
        if corr > best_corr:
            best_corr = corr
            best_tvt  = tw_tvt_local[i + half]
    return best_tvt, best_corr

def make_feats(h, tw):
    df = h.copy()
    ps = ps_idx(df)
    last_tvt = df.loc[ps, 'TVT_input']
    known    = df.loc[:ps, ['MD','TVT_input']].dropna()

    # dip ensemble
    dips = {}
    for tail_n, key in [(20,'dip20'),(50,'dip50'),(100,'dip100')]:
        tail = known.tail(tail_n)
        dips[key] = np.polyfit(tail['MD'], tail['TVT_input'], 1)[0] if len(tail)>2 else 0.
    dip_mean = np.mean(list(dips.values()))

    df['dmd_ps'] = df['MD'] - df.loc[ps,'MD']
    df['tvt_extrap_mean'] = last_tvt + dip_mean * df['dmd_ps']
    df['dip20'] = dips['dip20']; df['dip50'] = dips['dip50']; df['dip100'] = dips['dip100']
    df['dip_spread'] = max(dips.values()) - min(dips.values())

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

    df['tvt_filled']    = df['TVT_input'].interpolate().bfill().ffill()
    df['tvt_last']      = last_tvt
    df['steps_from_ps'] = (np.arange(len(df)) - ps).clip(min=0)
    df['dMD'] = df['MD'].diff().fillna(0)

    tw_s = tw.sort_values('TVT').dropna(subset=['TVT','GR'])
    tw_gr_arr  = tw_s['GR'].values
    tw_tvt_arr = tw_s['TVT'].values

    df['tw_gr']      = np.interp(df['tvt_filled'].values, tw_tvt_arr, tw_gr_arr)
    df['tw_residual']= df['gr'] - df['tw_gr']

    # ── Cross-correlation TVT correction ──────────────────────
    xcorr_tvt  = np.full(len(df), last_tvt)
    xcorr_corr = np.zeros(len(df))
    WIN = 20  # GR window size
    gr_vals = gr.values
    current_tvt_est = last_tvt

    for i in range(ps+1, len(df)):
        if i < WIN: continue
        gr_window = gr_vals[i-WIN:i]
        if np.isnan(gr_window).any(): continue
        est, corr = xcorr_tvt_estimate(gr_window, tw_gr_arr, tw_tvt_arr,
                                        current_tvt_est, search_range=40)
        xcorr_tvt[i]  = est
        xcorr_corr[i] = corr
        # EMA smoothing of TVT estimate
        current_tvt_est = 0.7*current_tvt_est + 0.3*est

    df['xcorr_tvt']   = xcorr_tvt
    df['xcorr_corr']  = xcorr_corr
    df['xcorr_delta'] = xcorr_tvt - df['tvt_extrap_mean']  # correction from linear extrap

    base = ['MD','dMD','dmd_ps','steps_from_ps','gr','gr_d1','gr_d2','gr_zscore',
            'gr_rm5','gr_rs5','gr_rm10','gr_rs10','gr_rm20','gr_rs20',
            'gr_rm40','gr_rs40','gr_rm80','gr_rs80',
            'tvt_filled','tvt_last','tw_gr','tw_residual',
            'dip20','dip50','dip100','dip_spread',
            'tvt_extrap_mean',
            'xcorr_tvt','xcorr_corr','xcorr_delta']
    xyz = [c for c in ['X','Y','Z'] if c in df.columns]

    if 'TVT' in df.columns:
        df['tvt_resid'] = df['TVT'] - df['tvt_extrap_mean']

    return df, base + xyz

rmse = lambda a, b: np.sqrt(mean_squared_error(a,b))

# ── split ─────────────────────────────────────────────────────
print('Loading wells …')
all_w = load_wells(TRAIN)
if not all_w:
    print(f"ERROR: No wells found in {TRAIN}")
    print(f"  Looking for files matching: *__horizontal_well.csv")
    print(f"  Please ensure TRAIN path is correct and data files exist.")
    raise ValueError(f"No wells loaded from {TRAIN}")

# Stratify by TVT band (same as Run 2)
meta = pd.DataFrame([{'wid':wid,'med_tvt':h['TVT'].median()} for wid,h,_ in all_w])
meta['band'] = pd.qcut(meta['med_tvt'], q=5, labels=False)
meta['fold'] = -1
for band, grp in meta.groupby('band'):
    shuf = grp.sample(frac=1, random_state=SEED)
    for i, idx in enumerate(shuf.index):
        meta.loc[idx,'fold'] = i % 5

fold_map  = dict(zip(meta['wid'], meta['fold']))
well_lookup = {wid:(h,tw) for wid,h,tw in all_w}

N_FOLD = 5
oof_true, oof_pred, oof_extrap = [], [], []
models, meds, fcolss = [], [], []

def make_ds(ids):
    Xs, ys = [], []
    fc = None
    for wid in ids:
        h,tw = well_lookup[wid]
        ps = ps_idx(h)
        df, fc_ = make_feats(h, tw)
        fc = fc_
        zone = df.iloc[ps+1:]
        if len(zone)==0: continue
        Xs.append(zone[fc_].values)
        ys.append(zone['tvt_resid'].values)
    X = np.vstack(Xs); y = np.concatenate(ys)
    med_ = np.nanmedian(X,axis=0)
    for j in range(X.shape[1]): X[np.isnan(X[:,j]),j]=med_[j]
    return X, y, fc, med_

lgb_params = dict(n_estimators=1200, learning_rate=0.025, max_depth=6,
                  num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                  min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=SEED, verbose=-1, n_jobs=-1)

for fold in range(N_FOLD):
    tr_ids  = meta[meta['fold']!=fold]['wid'].tolist()
    val_ids = meta[meta['fold']==fold]['wid'].tolist()
    print(f'\nFold {fold}: train={len(tr_ids)} val={len(val_ids)}')

    X_tr, y_tr, fcols, med_ = make_ds(tr_ids)
    X_vl, y_vl, _,     _    = make_ds(val_ids)
    for j in range(X_vl.shape[1]): X_vl[np.isnan(X_vl[:,j]),j]=med_[j]

    m = LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_vl,y_vl)],
          callbacks=[early_stopping(60,verbose=False), log_evaluation(200)])
    print(f'  Best iter: {m.best_iteration_}')

    for wid in val_ids:
        h,tw = well_lookup[wid]
        ps = ps_idx(h)
        df, fc = make_feats(h,tw)
        zone = df.iloc[ps+1:]
        if len(zone)==0: continue
        X_w = zone[fc].values.copy()
        for j in range(X_w.shape[1]): X_w[np.isnan(X_w[:,j]),j]=med_[j]
        oof_pred.extend(m.predict(X_w))
        oof_extrap.extend(zone['tvt_extrap_mean'].values)
        oof_true.extend(zone['TVT'].values)

    models.append(m); meds.append(med_); fcolss.append(fcols)

oof_tvt = np.array(oof_extrap) + np.array(oof_pred)
print(f'\nOOF RMSE: {rmse(np.array(oof_true), oof_tvt):.4f} ft')

# Feature importance (fold 0)
fi = pd.Series(models[0].feature_importances_, index=fcolss[0]).sort_values(ascending=False)
print('\nTop 20 features (fold 0):')
print(fi.head(20).to_string())

# ── inference ─────────────────────────────────────────────────
sample_sub = pd.read_csv(SUB)
sample_sub[['well_hash','md_int']] = sample_sub['id'].str.rsplit('_',n=1,expand=True)
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
    pred = zone['tvt_extrap_mean'].values + np.mean(fold_preds, axis=0)
    wh = wid[:8]
    sub_idx = sample_sub[sample_sub['well_hash']==wh].sort_values('md_int').index
    min_len = min(len(sub_idx), len(pred))
    sample_sub.loc[sub_idx[:min_len],'tvt'] = pred[:min_len]

sample_sub[['id','tvt']].to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print(f'\nRun 3 done. Submission saved to {OUTPUT_DIR / "submission.csv"}')
print(f'  {len(sample_sub)} rows')
print(sample_sub['tvt'].describe())
