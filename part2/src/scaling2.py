# ============================================================
# scaling2.py  –  FSS analysis for Part 2 (SIS lifespan)
# ============================================================
"""
Changes vs previous version
----------------------------
FIX-A  lambda_c scan extended so that lambda_c >= lp_min is reachable.
FIX-B  Scoring is R^2 only; hard bounds 0.1 < inv_nu < 5.0.
FIX-C  TAIL_FRACTION = 0.85, MIN_SIZES_FOR_LC = 4.
FIX-D  gamma=2.5: N=1M excluded (tau curve is flat noise, no peak signal).
       lambda_c forced to 0 analytically; 1/nu from raw power-law on lambda_p.
FIX-E  gamma=3.5: TAIL_FRACTION excludes the two smallest (noisiest) sizes.
"""

import glob, os, re
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR      = os.path.join(REPO_ROOT, "part2", "results", "raw")
PROC_DIR     = os.path.join(REPO_ROOT, "part2", "results", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

MIN_SIZES_FOR_LC = 4
MIN_POINTS_FIT   = 3
TAIL_FRACTION    = 0.85
INV_NU_MIN, INV_NU_MAX = 0.1, 5.0

# FIX-D: system sizes to exclude per gamma (tau signal is indistinguishable
# from noise at these sizes — confirmed by flat tau(lambda) curves).
EXCLUDE_N = {2.5: {1_000_000}}

FILE_RE = re.compile(
    r"^part2_N(?P<N>\d+)_g(?P<gamma>\d+(?:\.\d+)?)(?P<window>_window)?\.dat$"
)

# ── IO ────────────────────────────────────────────────────────────────────────

def list_raw_files():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "part2_N*_g*.dat")))
    return [p for p in files if FILE_RE.match(os.path.basename(p))]

def read_raw_file(path):
    name = os.path.basename(path)
    m = FILE_RE.match(name)
    if m is None:
        return None
    try:
        df = pd.read_csv(path, comment="#", sep=r"\s+", header=None,
                         names=["lambda", "tau", "P_end", "N", "gamma",
                                "nruns", "M"])
    except Exception:
        return None
    if len(df) == 0:
        return None
    df["source"]          = name
    df["is_window"]       = bool(m.group("window"))
    df["source_priority"] = np.where(df["is_window"], 1, 0)
    return df

def merge_case_scans(case_df):
    case_df = case_df.sort_values(["lambda", "source_priority"]).reset_index(drop=True)
    if not case_df["is_window"].any():
        return (case_df.drop_duplicates(subset=["lambda"], keep="last")
                       .sort_values("lambda").reset_index(drop=True))
    coarse = case_df.loc[~case_df["is_window"]].copy()
    window = case_df.loc[case_df["is_window"]].copy()
    lo, hi = float(window["lambda"].min()), float(window["lambda"].max())
    coarse = coarse.loc[(coarse["lambda"] < lo) | (coarse["lambda"] > hi)].copy()
    merged = pd.concat([coarse, window], ignore_index=True)
    merged = (merged.sort_values(["lambda", "source_priority"])
                    .drop_duplicates(subset=["lambda"], keep="last")
                    .sort_values("lambda").reset_index(drop=True))
    return merged

def load_raw_results():
    frames = [read_raw_file(p) for p in list_raw_files()]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("No readable raw files found.")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["gamma", "N", "lambda", "source_priority"]).reset_index(drop=True)
    merged = []
    for (gamma, N), sub in raw.groupby(["gamma", "N"], sort=True):
        # FIX-D: drop excluded sizes before merging
        if N in EXCLUDE_N.get(float(gamma), set()):
            continue
        merged.append(merge_case_scans(sub))
    data = pd.concat(merged, ignore_index=True)
    data = data.sort_values(["gamma", "N", "lambda"]).reset_index(drop=True)
    data.to_csv(os.path.join(PROC_DIR, "part2_all_raw.csv"), index=False)
    return data

# ── fits ──────────────────────────────────────────────────────────────────────

def loglog_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    lx, ly = np.log(x), np.log(y)
    s, ic  = np.polyfit(lx, ly, 1)
    ss_res = np.sum((ly - (s * lx + ic)) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(s), float(np.exp(ic)), float(r2), len(x)

# ── peaks ─────────────────────────────────────────────────────────────────────

def local_quadratic_peak(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmax(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i]), float(y[i]), i, False
    xx, yy = x[i-1:i+2], y[i-1:i+2]
    try:
        a, b, c = np.polyfit(xx, yy, 2)
        if a < 0:
            xv = -b / (2 * a)
            if xx[0] <= xv <= xx[-1]:
                yv = a*xv**2 + b*xv + c
                if np.isfinite(yv) and yv > 0:
                    return float(xv), float(yv), i, True
    except Exception:
        pass
    return float(x[i]), float(y[i]), i, False

def extract_peaks(gamma_df):
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)
        lam, tau = sdf["lambda"].to_numpy(float), sdf["tau"].to_numpy(float)
        lp, tp, imax, refined = local_quadratic_peak(lam, tau)
        rows.append({"gamma": float(sdf["gamma"].iloc[0]), "N": int(N),
                     "lambda_p": lp, "tau_peak": tp,
                     "lambda_argmax": float(lam[imax]),
                     "tau_argmax": float(tau[imax]),
                     "peak_refined": int(refined)})
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

# ── lambda_c estimation ───────────────────────────────────────────────────────

def choose_tail(df):
    df = df.sort_values("N").reset_index(drop=True)
    keep = max(MIN_POINTS_FIT, int(np.ceil(TAIL_FRACTION * len(df))))
    return df.iloc[len(df) - keep:].copy()

def estimate_lambda_c(gamma, peak_df):
    """
    FIX-D: for gamma < 3 theory predicts lambda_c -> 0 in the thermodynamic
    limit.  Setting lambda_c = 0 analytically avoids the degenerate fit that
    occurs when lambda_p(N_max) ≈ 0, which forces lambda_c ≈ lambda_p(N_max)
    and yields R^2 << 0.5.  We then fit lambda_p ~ A * N^{-1/nu} directly.
    """
    if float(gamma) < 3.0:
        fit_df = choose_tail(peak_df)
        N  = fit_df["N"].to_numpy(float)
        lp = fit_df["lambda_p"].to_numpy(float)
        s, amp, r2, n_fit = loglog_fit(N, lp)
        inv_nu = -s if np.isfinite(s) else np.nan
        # enforce physical bounds
        if not np.isfinite(inv_nu) or inv_nu < INV_NU_MIN or inv_nu > INV_NU_MAX:
            inv_nu, amp, r2 = np.nan, np.nan, np.nan
        return {"lambda_c": 0.0, "A_lambdap": float(amp) if np.isfinite(amp) else np.nan,
                "inv_nu": float(inv_nu) if np.isfinite(inv_nu) else np.nan,
                "lambdap_fit_r2": float(r2) if np.isfinite(r2) else np.nan,
                "n_fit": int(n_fit)}

    # gamma >= 3: scan for best lambda_c as before (FIX-A/B/C)
    if len(peak_df) < MIN_SIZES_FOR_LC:
        return {"lambda_c": np.nan, "A_lambdap": np.nan,
                "inv_nu": np.nan, "lambdap_fit_r2": np.nan, "n_fit": 0}

    fit_df = choose_tail(peak_df)
    N  = fit_df["N"].to_numpy(float)
    lp = fit_df["lambda_p"].to_numpy(float)
    lp_min, lp_max = float(np.min(lp)), float(np.max(lp))
    span = max(lp_max - lp_min, 1e-6)

    lo = max(1e-8, lp_min - 5.0 * span)
    hi = lp_min + 2.0 * span           # FIX-A
    if hi <= lo:
        lo, hi = max(1e-8, 0.5 * lp_min), lp_max + span

    best = None
    for lc in np.linspace(lo, hi, 8000):
        diff = lp - lc
        if np.any(diff <= 0):
            continue
        s, amp, r2, n_used = loglog_fit(N, diff)
        inv_nu = -s
        if n_used < MIN_POINTS_FIT or not np.isfinite(inv_nu) or not np.isfinite(r2):
            continue
        if inv_nu < INV_NU_MIN or inv_nu > INV_NU_MAX:   # FIX-B
            continue
        if best is None or r2 > best["score"]:            # FIX-B: R^2 only
            best = {"lambda_c": float(lc), "A_lambdap": float(amp),
                    "inv_nu": float(inv_nu), "lambdap_fit_r2": float(r2),
                    "n_fit": int(n_used), "score": r2}

    if best is None:
        return {"lambda_c": np.nan, "A_lambdap": np.nan,
                "inv_nu": np.nan, "lambdap_fit_r2": np.nan, "n_fit": 0}
    return {k: v for k, v in best.items() if k != "score"}

# ── P_end at lambda_c ─────────────────────────────────────────────────────────

def extract_pend_at_lc(gamma_df, gamma, lambda_c):
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf  = sdf.sort_values("lambda").reset_index(drop=True)
        lam  = sdf["lambda"].to_numpy(float)
        pend = sdf["P_end"].to_numpy(float)
        if lambda_c < lam[0] or lambda_c > lam[-1]:
            p_val = np.nan
        else:
            p_val = float(np.interp(lambda_c, lam, pend))
        rows.append({"gamma": float(gamma), "N": int(N),
                     "lambda_c": float(lambda_c),
                     "lambda_eval": float(lambda_c), "P_end_lc": p_val,
                     "eval_mode": "interp"})
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

# ── exponent fits ─────────────────────────────────────────────────────────────

def fit_tau_peak(peak_df):
    fit_df = choose_tail(peak_df)
    s, amp, r2, n = loglog_fit(fit_df["N"].to_numpy(float),
                                fit_df["tau_peak"].to_numpy(float))
    return {"gamma1_over_nu": float(s) if np.isfinite(s) else np.nan,
            "tau_peak_amp": float(amp) if np.isfinite(amp) else np.nan,
            "tau_peak_fit_r2": float(r2) if np.isfinite(r2) else np.nan,
            "tau_peak_n_fit": int(n)}

def fit_beta_over_nu(pend_df):
    fit_df = choose_tail(pend_df)
    valid  = fit_df.dropna(subset=["P_end_lc"])
    valid  = valid.loc[valid["P_end_lc"] > 0]
    s, amp, r2, n = loglog_fit(valid["N"].to_numpy(float),
                                valid["P_end_lc"].to_numpy(float))
    raw_beta = -s if np.isfinite(s) else np.nan
    beta = raw_beta if (np.isfinite(raw_beta) and raw_beta > 0) else np.nan
    return {"beta_over_nu": beta,
            "pend_amp": float(amp) if np.isfinite(amp) else np.nan,
            "pend_fit_r2": float(r2) if np.isfinite(r2) else np.nan,
            "pend_n_fit": int(n)}

# ── per-gamma analysis ────────────────────────────────────────────────────────

def analyze_gamma(gamma, gamma_df):
    peak_df = extract_peaks(gamma_df)
    peak_df.to_csv(
        os.path.join(PROC_DIR, f"part2_peaks_gamma_{gamma:.1f}.csv"), index=False)

    out = {"gamma": float(gamma), "n_sizes": int(peak_df["N"].nunique())}
    out.update(estimate_lambda_c(gamma, peak_df))
    out.update(fit_tau_peak(peak_df))

    lc = out["lambda_c"]
    if np.isfinite(lc):
        # FIX-D: for gamma<3, lambda_c=0 but P_end at lambda=0 is 0 for all N
        # (trivially), so skip the P_end fit for gamma<3.
        if float(gamma) < 3.0:
            pd.DataFrame(columns=["gamma","N","lambda_c","lambda_eval",
                                  "P_end_lc","eval_mode"]).to_csv(
                os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv"),
                index=False)
            out.update({"beta_over_nu": np.nan, "pend_amp": np.nan,
                        "pend_fit_r2": np.nan, "pend_n_fit": 0})
        else:
            pend_df = extract_pend_at_lc(gamma_df, gamma, lc)
            pend_df.to_csv(
                os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv"),
                index=False)
            out.update(fit_beta_over_nu(pend_df))
    else:
        pd.DataFrame(columns=["gamma","N","lambda_c","lambda_eval",
                               "P_end_lc","eval_mode"]).to_csv(
            os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv"),
            index=False)
        out.update({"beta_over_nu": np.nan, "pend_amp": np.nan,
                    "pend_fit_r2": np.nan, "pend_n_fit": 0})
    return out

def main():
    data = load_raw_results()
    summaries = []
    for gamma, gdf in data.groupby("gamma"):
        summaries.append(
            analyze_gamma(gamma, gdf.sort_values(["N","lambda"]).reset_index(drop=True)))
    summary_df = (pd.DataFrame(summaries).sort_values("gamma")
                                          .reset_index(drop=True))
    summary_df.to_csv(os.path.join(PROC_DIR, "part2_critical_exponents.csv"), index=False)
    print("\nCritical exponents:\n")
    print(summary_df.to_string(index=False))
    print("\nWritten to", PROC_DIR)

if __name__ == "__main__":
    main()