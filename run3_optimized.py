
# Optimized Run 3 — Typewell cross-correlation TVT correction feature
# Includes:
# - Timing metrics
# - Feature caching
# - Vectorized NaN filling
# - Faster cross-correlation
# - Reduced repeated feature computation

import time
import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# TIMER
# ============================================================

class Timer:
    def __init__(self):
        self.times = {}

    def start(self, name):
        self.times[name] = self.times.get(name, 0.0) - time.perf_counter()

    def stop(self, name):
        self.times[name] += time.perf_counter()

    def report(self):
        print("\n" + "=" * 80)
        print("TIMING REPORT")
        print("=" * 80)

        total = 0.0
        for k, v in sorted(self.times.items(), key=lambda x: -x[1]):
            if v > 0:
                total += v
                print(f"{k:<35} {v:10.2f} sec")

        print("-" * 80)
        print(f"{'TOTAL':<35} {total:10.2f} sec")
        print("=" * 80)


timer = Timer()
timer.start("total")

# ============================================================
# PATHS
# ============================================================

timer.start("loading")

if os.path.exists("/kaggle/input"):
    kaggle_input = Path("/kaggle/input")
    train_dirs = list(kaggle_input.rglob("train"))

    if train_dirs:
        BASE = train_dirs[0].parent
    else:
        BASE = Path("/kaggle/input/rogii-wellbore-geology-prediction")

    TRAIN = BASE / "train"
    TEST = BASE / "test"
    SUB = BASE / "sample_submission.csv"
    OUTPUT_DIR = Path("/kaggle/working")

else:
    BASE = Path.cwd()

    if not (BASE / "train").exists():
        if (BASE.parent / "train").exists():
            BASE = BASE.parent

    TRAIN = BASE / "train"
    TEST = BASE / "test"
    SUB = BASE / "sample_submission.csv"
    OUTPUT_DIR = BASE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ============================================================
# HELPERS
# ============================================================

def load_wells(d):
    out = []

    for hf in sorted(Path(d).glob("*__horizontal_well.csv")):
        wid = hf.stem.replace("__horizontal_well", "")

        tws = list(Path(d).glob(f"{wid}__typewell.csv"))

        if not tws:
            continue

        out.append(
            (
                wid,
                pd.read_csv(hf),
                pd.read_csv(tws[0]),
            )
        )

    return out


def ps_idx(h):
    v = h["TVT_input"].notna()
    return int(v[v].index[-1]) if v.any() else 0


def fast_nan_fill(X, med):
    inds = np.where(np.isnan(X))
    if len(inds[0]):
        X[inds] = np.take(med, inds[1])
    return X


def xcorr_tvt_estimate(
    gr_window,
    tw_gr_arr,
    tw_tvt_arr,
    center_tvt,
    search_range=40,
):

    mask = np.abs(tw_tvt_arr - center_tvt) < search_range

    if mask.sum() < len(gr_window) + 2:
        return center_tvt, 0.0

    tw_gr_local = tw_gr_arr[mask]
    tw_tvt_local = tw_tvt_arr[mask]

    W = len(gr_window)

    if len(tw_gr_local) <= W:
        return center_tvt, 0.0

    gw = gr_window.astype(np.float32)
    gw = gw - gw.mean()

    gw_std = gw.std()

    if gw_std < 1e-6:
        return center_tvt, 0.0

    gw /= gw_std

    windows = np.lib.stride_tricks.sliding_window_view(
        tw_gr_local,
        W
    )

    win_mean = windows.mean(axis=1, keepdims=True)
    win_std = windows.std(axis=1, keepdims=True) + 1e-6

    windows = (windows - win_mean) / win_std

    corr = (windows * gw).mean(axis=1)

    best_idx = np.argmax(corr)

    best_corr = float(corr[best_idx])

    center_idx = best_idx + W // 2

    return tw_tvt_local[center_idx], best_corr


# ============================================================
# FEATURES
# ============================================================

def make_feats(h, tw):

    df = h.copy()

    ps = ps_idx(df)

    last_tvt = df.loc[ps, "TVT_input"]

    known = df.loc[:ps, ["MD", "TVT_input"]].dropna()

    dips = {}

    for tail_n, key in [
        (20, "dip20"),
        (50, "dip50"),
        (100, "dip100"),
    ]:
        tail = known.tail(tail_n)

        dips[key] = (
            np.polyfit(
                tail["MD"],
                tail["TVT_input"],
                1,
            )[0]
            if len(tail) > 2
            else 0.0
        )

    dip_mean = np.mean(list(dips.values()))

    df["dmd_ps"] = df["MD"] - df.loc[ps, "MD"]

    df["tvt_extrap_mean"] = (
        last_tvt + dip_mean * df["dmd_ps"]
    )

    df["dip20"] = dips["dip20"]
    df["dip50"] = dips["dip50"]
    df["dip100"] = dips["dip100"]

    df["dip_spread"] = (
        max(dips.values()) - min(dips.values())
    )

    gr = (
        df["GR"]
        .interpolate()
        .bfill()
        .ffill()
    )

    df["gr"] = gr

    for w in [5, 10, 20, 40, 80]:
        df[f"gr_rm{w}"] = gr.rolling(
            w,
            min_periods=1
        ).mean()

        df[f"gr_rs{w}"] = (
            gr.rolling(
                w,
                min_periods=1
            )
            .std()
            .fillna(0)
        )

    df["gr_d1"] = gr.diff().fillna(0)
    df["gr_d2"] = df["gr_d1"].diff().fillna(0)

    gr_mu = (
        gr.iloc[: ps + 1]
        .tail(50)
        .mean()
    )

    gr_sig = (
        gr.iloc[: ps + 1]
        .tail(50)
        .std()
        + 1e-6
    )

    df["gr_zscore"] = (
        gr - gr_mu
    ) / gr_sig

    df["tvt_filled"] = (
        df["TVT_input"]
        .interpolate()
        .bfill()
        .ffill()
    )

    df["tvt_last"] = last_tvt

    df["steps_from_ps"] = (
        np.arange(len(df)) - ps
    ).clip(min=0)

    df["dMD"] = df["MD"].diff().fillna(0)

    tw_s = (
        tw.sort_values("TVT")
        .dropna(
            subset=["TVT", "GR"]
        )
    )

    tw_gr_arr = tw_s["GR"].values
    tw_tvt_arr = tw_s["TVT"].values

    df["tw_gr"] = np.interp(
        df["tvt_filled"].values,
        tw_tvt_arr,
        tw_gr_arr,
    )

    df["tw_residual"] = (
        df["gr"] - df["tw_gr"]
    )

    xcorr_tvt = np.full(
        len(df),
        last_tvt,
        dtype=np.float32,
    )

    xcorr_corr = np.zeros(
        len(df),
        dtype=np.float32,
    )

    WIN = 20

    gr_vals = gr.values

    current_tvt_est = last_tvt

    for i in range(ps + 1, len(df)):

        if i < WIN:
            continue

        gr_window = gr_vals[i - WIN : i]

        est, corr = xcorr_tvt_estimate(
            gr_window,
            tw_gr_arr,
            tw_tvt_arr,
            current_tvt_est,
            search_range=40,
        )

        xcorr_tvt[i] = est
        xcorr_corr[i] = corr

        current_tvt_est = (
            0.7 * current_tvt_est
            + 0.3 * est
        )

    df["xcorr_tvt"] = xcorr_tvt
    df["xcorr_corr"] = xcorr_corr
    df["xcorr_delta"] = (
        xcorr_tvt
        - df["tvt_extrap_mean"]
    )

    base = [
        "MD","dMD","dmd_ps","steps_from_ps",
        "gr","gr_d1","gr_d2","gr_zscore",
        "gr_rm5","gr_rs5",
        "gr_rm10","gr_rs10",
        "gr_rm20","gr_rs20",
        "gr_rm40","gr_rs40",
        "gr_rm80","gr_rs80",
        "tvt_filled","tvt_last",
        "tw_gr","tw_residual",
        "dip20","dip50","dip100",
        "dip_spread",
        "tvt_extrap_mean",
        "xcorr_tvt",
        "xcorr_corr",
        "xcorr_delta",
    ]

    xyz = [
        c
        for c in ["X", "Y", "Z"]
        if c in df.columns
    ]

    if "TVT" in df.columns:
        df["tvt_resid"] = (
            df["TVT"]
            - df["tvt_extrap_mean"]
        )

    return df, base + xyz


rmse = lambda a, b: np.sqrt(
    mean_squared_error(a, b)
)

# ============================================================
# LOAD WELLS
# ============================================================

all_w = load_wells(TRAIN)

timer.stop("loading")

print(f"Loaded {len(all_w)} wells")

# ============================================================
# FEATURE CACHE
# ============================================================

timer.start("feature_engineering")

feature_cache = {}

for wid, h, tw in all_w:

    feature_df, feat_cols = make_feats(
        h,
        tw,
    )

    feature_cache[wid] = {
        "df": feature_df,
        "feat_cols": feat_cols,
        "ps": ps_idx(h),
    }

timer.stop("feature_engineering")

meta = pd.DataFrame(
    [
        {
            "wid": wid,
            "med_tvt": h["TVT"].median(),
        }
        for wid, h, _ in all_w
    ]
)

meta["band"] = pd.qcut(
    meta["med_tvt"],
    q=5,
    labels=False,
)

meta["fold"] = -1

for band, grp in meta.groupby("band"):

    shuf = grp.sample(
        frac=1,
        random_state=SEED,
    )

    for i, idx in enumerate(shuf.index):
        meta.loc[idx, "fold"] = i % 5

N_FOLD = 5

def make_ds(ids):

    Xs = []
    ys = []

    feat_cols = None

    for wid in ids:

        cache = feature_cache[wid]

        df = cache["df"]
        ps = cache["ps"]

        feat_cols = cache["feat_cols"]

        zone = df.iloc[ps + 1 :]

        if len(zone) == 0:
            continue

        Xs.append(
            zone[feat_cols]
            .to_numpy(np.float32)
        )

        ys.append(
            zone["tvt_resid"]
            .to_numpy(np.float32)
        )

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    med = np.nanmedian(
        X,
        axis=0,
    )

    X = fast_nan_fill(X, med)

    return X, y, feat_cols, med

print("Optimized script generated successfully.")
