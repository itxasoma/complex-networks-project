# ============================================================
# scaling2.py  –  Assignment analysis + diagnostics
# Changes vs previous version:
#   - estimate_lambda_c (gamma>=3): monotonicity-violation filter removes
#     peaks that jump UP with increasing N before the lambda_c grid search.
#     This eliminates the N=30k spurious peak for gamma=3.5.
#   - fit_tau_peak: same monotonicity filter so the N=30k outlier
#     does not corrupt gamma1/nu.
#   - extract_pend_at_lc / fit_beta_over_nu: keep P_end_lc=0 values
#     (flagged as interp_zero) and attempt the fit honestly; if no
#     positive P_end values exist at lambda_c, beta/nu is reported as NaN
#     with a clear reason string instead of silently dropping the step.
#   - check_step5_feasibility: added 'beta_nu_reason' field explaining
#     WHY beta/nu cannot be extracted when that is the case.
# ============================================================

import glob
import os
import re
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR    = os.path.join(REPO_ROOT, "part2", "results", "raw")
PROC_DIR   = os.path.join(REPO_ROOT, "part2", "results", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


MIN_SIZES_FOR_LC  = 4
MIN_POINTS_FIT    = 3
TAIL_FRACTION     = 0.85
INV_NU_MIN, INV_NU_MAX = 0.05, 5.0

# Hard exclusions: N=1M for gamma=2.5 (scan does not reach the critical region).
EXCLUDE_N = {2.5: {1_000_000}}

FILE_RE = re.compile(
    r"^part2_N(?P<N>\d+)_g(?P<gamma>\d+(?:\.\d+)?)(?P<variant>_lc_window|_window\d*|_window)?"
    r"(?P<r>_r\d+)?"
    r"\.dat$"
)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def list_raw_files():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "part2_N*_g*.dat")))
    return [p for p in files if FILE_RE.match(os.path.basename(p))]


def read_raw_file(path):
    name = os.path.basename(path)
    m = FILE_RE.match(name)
    if m is None:
        return None
    try:
        df = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+",
            header=None,
            names=["lambda", "tau", "P_end", "N", "gamma", "nruns", "M"]
        )
    except Exception:
        return None
    if len(df) == 0:
        return None
    variant = m.group("variant") or ""
    if variant == "_lc_window":
        prio = 3
    elif variant.startswith("_window"):
        if variant == "_window2":
            prio = 2
        else:
            prio = 1
    else:
        prio = 0

    df["source"]          = name
    df["variant"]         = variant if variant else "coarse"
    df["is_window"]       = prio > 0
    df["source_priority"] = prio
    return df


def merge_case_scans(case_df):
    """
    Merge overlapping scans for one (gamma, N).

    Rules:
      - read coarse, window, window2, lc_window
      - in overlapping lambda regions, keep only the most refined tier
      - if the same lambda appears multiple times within the surviving tier,
        average tau and P_end instead of keeping the last row
    """
    case_df = case_df.sort_values(["lambda", "source_priority"]).reset_index(drop=True)

    def average_same_lambda(df):
        if len(df) == 0:
            return df.copy()
        agg = {
            "tau": "mean",
            "P_end": "mean",
            "gamma": "first",
            "N": "first",
            "nruns": "max",
            "M": "max",
            "source_priority": "first",
            "is_window": "first",
            "variant": lambda s: ";".join(sorted(set(map(str, s)))),
            "source": lambda s: ";".join(sorted(set(map(str, s)))),
        }
        return (
            df.groupby("lambda", as_index=False)
              .agg(agg)
              .sort_values("lambda")
              .reset_index(drop=True)
        )

    if not case_df["is_window"].any():
        return average_same_lambda(case_df)

    pieces = []

    # Apply from most refined to least refined
    priorities = sorted(case_df["source_priority"].unique(), reverse=True)
    for prio in priorities:
        tier = case_df.loc[case_df["source_priority"] == prio].copy()
        if len(tier) == 0:
            continue

        if prio > 0:
            lo, hi = float(tier["lambda"].min()), float(tier["lambda"].max())
            mask = (case_df["lambda"] >= lo) & (case_df["lambda"] <= hi)
            case_df = case_df.loc[~(mask & (case_df["source_priority"] < prio))].copy()
            tier = case_df.loc[case_df["source_priority"] == prio].copy()

        pieces.append(average_same_lambda(tier))
        case_df = case_df.loc[case_df["source_priority"] != prio].copy()

    merged = pd.concat(pieces, ignore_index=True)
    merged = average_same_lambda(merged)
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
        if N in EXCLUDE_N.get(float(gamma), set()):
            continue
        merged.append(merge_case_scans(sub))

    data = pd.concat(merged, ignore_index=True)
    data = data.sort_values(["gamma", "N", "lambda"]).reset_index(drop=True)
    data.to_csv(os.path.join(PROC_DIR, "part2_all_raw.csv"), index=False)
    return data


# ---------------------------------------------------------------------------
# Basic fits
# ---------------------------------------------------------------------------

def loglog_fit(x, y):
    """OLS log-log fit: y ~ A * x^s. Returns (s, A, R2, n_points)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)

    lx, ly = np.log(x), np.log(y)
    s, ic  = np.polyfit(lx, ly, 1)

    yhat   = s * lx + ic
    ss_res = np.sum((ly - yhat) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2     = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return float(s), float(np.exp(ic)), float(r2), len(x)


def choose_tail(df):
    """Keep the largest-N fraction of rows (at least MIN_POINTS_FIT)."""
    df   = df.sort_values("N").reset_index(drop=True)
    keep = max(MIN_POINTS_FIT, int(np.ceil(TAIL_FRACTION * len(df))))
    return df.iloc[len(df) - keep:].copy()


# ---------------------------------------------------------------------------
# Outlier filter for peak sequences
# ---------------------------------------------------------------------------

def monotone_filter(N_arr, val_arr, direction="decreasing"):
    """
    Remove peaks whose value *increases* with N by more than 1 IQR
    (gross monotonicity violations — local upward spikes).

    A point is flagged if:
      - it is an interior point with val[i] > val[i-1] + IQR  AND  val[i] > val[i+1]
      - OR it is the second point (i=1), jumps from val[0] by > IQR, and
        the sequence then resumes decreasing (val[i] > val[i+1])
      - OR it is the last point and creates a trailing upward jump > IQR

    Returns a boolean mask (True = keep).
    """
    N   = np.asarray(N_arr, float)
    v   = np.asarray(val_arr, float)
    n   = len(v)
    if n < 4:
        return np.ones(n, dtype=bool)

    keep = np.ones(n, dtype=bool)
    iqr  = np.percentile(v, 75) - np.percentile(v, 25)
    thr  = max(iqr, 1e-8)          # jump must exceed 1 IQR to be flagged

    for i in range(n):
        if i == 0:
            pass  # first point: never flag based on left neighbour alone
        elif i == n - 1:
            # trailing upward spike
            if direction == "decreasing" and v[i] > v[i - 1] + thr:
                keep[i] = False
        else:
            # interior spike: rises from previous AND falls to next
            if direction == "decreasing":
                if v[i] > v[i - 1] + thr and v[i] > v[i + 1]:
                    keep[i] = False
                elif i == 1 and v[i] > v[i - 1] + thr and v[i] > v[i + 1]:
                    keep[i] = False

    return keep


# ---------------------------------------------------------------------------
# Step 3: tau peaks
# ---------------------------------------------------------------------------

def local_quadratic_peak(x, y):
    """Refine the argmax with a local quadratic fit."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(np.argmax(y))

    if i == 0 or i == len(x) - 1:
        return float(x[i]), float(y[i]), i, False

    xx, yy = x[i - 1:i + 2], y[i - 1:i + 2]
    try:
        a, b, c = np.polyfit(xx, yy, 2)
        if a < 0:
            xv = -b / (2 * a)
            if xx[0] <= xv <= xx[-1]:
                yv = a * xv ** 2 + b * xv + c
                if np.isfinite(yv) and yv > 0:
                    return float(xv), float(yv), i, True
    except Exception:
        pass

    return float(x[i]), float(y[i]), i, False


def extract_tau_peaks(gamma_df):
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)
        lam = sdf["lambda"].to_numpy(float)
        tau = sdf["tau"].to_numpy(float)
        lp, tp, imax, refined = local_quadratic_peak(lam, tau)
        rows.append({
            "gamma":        float(sdf["gamma"].iloc[0]),
            "N":            int(N),
            "lambda_p":     lp,
            "tau_peak":     tp,
            "lambda_argmax": float(lam[imax]),
            "tau_argmax":   float(tau[imax]),
            "peak_refined": int(refined),
        })
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: estimate lambda_c and 1/nu from peak positions
# ---------------------------------------------------------------------------

def estimate_lambda_c(gamma, peak_df):
    """
    gamma < 3  (e.g. 2.5): lambda_c = 0 by theory; fit lambda_p ~ A*N^{-1/nu}.
    gamma >= 3 (e.g. 3.5): fit lambda_p(N) - lambda_c ~ A*N^{-1/nu} scanning
                            over lambda_c, after removing monotonicity-violating
                            peaks (local upward spikes in N that exceed 1 IQR).
    """
    if float(gamma) < 3.0:
        fit_df = choose_tail(peak_df)
        N   = fit_df["N"].to_numpy(float)
        lp  = fit_df["lambda_p"].to_numpy(float)

        s, amp, r2, n_fit = loglog_fit(N, lp)
        inv_nu = -s if np.isfinite(s) else np.nan

        if not np.isfinite(inv_nu) or inv_nu < INV_NU_MIN or inv_nu > INV_NU_MAX:
            inv_nu, amp, r2 = np.nan, np.nan, np.nan

        return {
            "lambda_c":        0.0,
            "A_lambdap":       float(amp)    if np.isfinite(amp)    else np.nan,
            "inv_nu":          float(inv_nu) if np.isfinite(inv_nu) else np.nan,
            "lambdap_fit_r2":  float(r2)     if np.isfinite(r2)     else np.nan,
            "n_fit":           int(n_fit),
            "n_outliers_removed": 0,
        }

    # --- gamma >= 3 path ---
    if len(peak_df) < MIN_SIZES_FOR_LC:
        return {
            "lambda_c": np.nan, "A_lambdap": np.nan, "inv_nu": np.nan,
            "lambdap_fit_r2": np.nan, "n_fit": 0, "n_outliers_removed": 0,
        }

    # Apply monotonicity filter FIRST (on the full set, not tail)
    all_df  = peak_df.sort_values("N").reset_index(drop=True)
    N_all   = all_df["N"].to_numpy(float)
    lp_all  = all_df["lambda_p"].to_numpy(float)
    keep    = monotone_filter(N_all, lp_all, direction="decreasing")
    n_removed = int((~keep).sum())

    # After removing outliers, apply tail selection
    clean_df = all_df[keep].reset_index(drop=True)
    if len(clean_df) < MIN_SIZES_FOR_LC:
        # Not enough clean points; fall back to full set
        clean_df   = all_df.copy()
        n_removed  = 0

    fit_df = choose_tail(clean_df).sort_values("N").reset_index(drop=True)
    N      = fit_df["N"].to_numpy(float)
    lp     = fit_df["lambda_p"].to_numpy(float)

    if len(N) < MIN_POINTS_FIT:
        return {
            "lambda_c": np.nan, "A_lambdap": np.nan, "inv_nu": np.nan,
            "lambdap_fit_r2": np.nan, "n_fit": len(N),
            "n_outliers_removed": n_removed,
        }

    lp_min = float(np.min(lp))
    lp_max = float(np.max(lp))
    span   = max(lp_max - lp_min, 1e-6)

    # Wide search range: lambda_c in [lp_min - 2*span, lp_min - eps]
    local_band = max(0.01, 2.0 * span)
    lo = max(1e-5, lp_min - local_band)
    hi = lp_min - 1e-5

    best = None
    for lc in np.linspace(lo, hi, 6000):
        diff = lp - lc
        if np.any(diff <= 0):
            continue

        s, amp, r2, n_used = loglog_fit(N, diff)
        inv_nu = -s

        if n_used < MIN_POINTS_FIT:
            continue
        if not np.isfinite(inv_nu) or not np.isfinite(r2):
            continue
        if inv_nu < INV_NU_MIN or inv_nu > INV_NU_MAX:
            continue

        cand = {
            "lambda_c":       float(lc),
            "A_lambdap":      float(amp),
            "inv_nu":         float(inv_nu),
            "lambdap_fit_r2": float(r2),
            "n_fit":          int(n_used),
        }

        if best is None:
            best = cand
        else:
            better_r2 = cand["lambdap_fit_r2"] > best["lambdap_fit_r2"] + 1e-12
            tie_r2    = abs(cand["lambdap_fit_r2"] - best["lambdap_fit_r2"]) <= 1e-12
            larger_lc = cand["lambda_c"] > best["lambda_c"]
            if better_r2 or (tie_r2 and larger_lc):
                best = cand

    if best is None:
        return {
            "lambda_c": np.nan, "A_lambdap": np.nan, "inv_nu": np.nan,
            "lambdap_fit_r2": np.nan, "n_fit": 0,
            "n_outliers_removed": n_removed,
        }

    best["n_outliers_removed"] = n_removed
    return best


# ---------------------------------------------------------------------------
# Step 4: tau peak height -> gamma1/nu
# ---------------------------------------------------------------------------

def fit_tau_peak(peak_df):
    """
    Fit tau_peak ~ A * N^{gamma1/nu}.
    Applies the same monotonicity filter as estimate_lambda_c so that
    spurious high-tau outliers (e.g. N=30k for gamma=3.5) do not
    corrupt the exponent.
    """
    df  = peak_df.sort_values("N").reset_index(drop=True)
    N   = df["N"].to_numpy(float)
    tp  = df["tau_peak"].to_numpy(float)

    keep = monotone_filter(N, tp, direction="decreasing")
    N_c  = N[keep]
    tp_c = tp[keep]
    n_removed_tp = int((~keep).sum())

    if len(N_c) < MIN_POINTS_FIT:
        N_c, tp_c = N, tp   # fall back to full set
        n_removed_tp = 0

    tmp    = pd.DataFrame({"N": N_c, "tau_peak": tp_c})
    fit_df = choose_tail(tmp)

    s, amp, r2, n = loglog_fit(
        fit_df["N"].to_numpy(float),
        fit_df["tau_peak"].to_numpy(float)
    )
    return {
        "gamma1_over_nu":     float(s)   if np.isfinite(s)   else np.nan,
        "tau_peak_amp":       float(amp) if np.isfinite(amp) else np.nan,
        "tau_peak_fit_r2":    float(r2)  if np.isfinite(r2)  else np.nan,
        "tau_peak_n_fit":     int(n),
        "tau_peak_n_removed": int(n_removed_tp),
    }


# ---------------------------------------------------------------------------
# Step 5: P_end(lambda_c, N) -> beta/nu
# ---------------------------------------------------------------------------

def extract_pend_at_lc(gamma_df, gamma, lambda_c):
    """
    Evaluate P_end at lambda_c for each N by linear interpolation.
    Keeps P_end_lc = 0 entries (flagged as 'interp_zero') instead of
    dropping them so fit_beta_over_nu can diagnose the situation properly.
    """
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf  = sdf.sort_values("lambda").reset_index(drop=True)
        lam  = sdf["lambda"].to_numpy(float)
        pend = sdf["P_end"].to_numpy(float)

        if lambda_c < lam[0]:
            p_val = np.nan
            mode  = "out_of_range_low"
        elif lambda_c > lam[-1]:
            p_val = np.nan
            mode  = "out_of_range_high"
        else:
            p_val = float(np.interp(lambda_c, lam, pend))
            mode  = "interp" if p_val > 0 else "interp_zero"

        rows.append({
            "gamma":       float(gamma),
            "N":           int(N),
            "lambda_c":    float(lambda_c),
            "lambda_eval": float(lambda_c),
            "P_end_lc":    float(p_val) if np.isfinite(p_val) else np.nan,
            "eval_mode":   mode,
        })

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def fit_beta_over_nu(pend_df):
    """
    Fit P_end(lambda_c, N) ~ A * N^{-beta/nu}.
    Only uses rows where P_end_lc > 0.
    Reports beta/nu = NaN with a descriptive reason if there are fewer
    than MIN_POINTS_FIT positive values.
    """
    valid = pend_df.dropna(subset=["P_end_lc"]).copy()
    valid = valid.loc[valid["P_end_lc"] > 0]

    if len(valid) < MIN_POINTS_FIT:
        n_zero = int((pend_df["eval_mode"] == "interp_zero").sum())
        n_oor  = int(pend_df["eval_mode"].str.startswith("out_of_range").sum())
        if n_zero > 0:
            reason = (
                f"P_end(lambda_c,N)=0 for all {n_zero} interpolated sizes: "
                "lambda_c lies below the onset of endemic probability in the "
                "simulated lambda range. The Fortran scan did not cover "
                "lambda values between 0 and the P_end onset."
            )
        elif n_oor > 0:
            reason = f"lambda_c out of simulated lambda range for {n_oor} sizes."
        else:
            reason = f"Fewer than {MIN_POINTS_FIT} sizes with P_end(lambda_c)>0."
        return {
            "beta_over_nu":   np.nan,
            "pend_amp":       np.nan,
            "pend_fit_r2":    np.nan,
            "pend_n_fit":     0,
            "beta_nu_reason": reason,
        }

    fit_df   = choose_tail(valid)
    s, amp, r2, n = loglog_fit(
        fit_df["N"].to_numpy(float),
        fit_df["P_end_lc"].to_numpy(float)
    )
    raw_beta = -s if np.isfinite(s) else np.nan
    beta     = raw_beta if (np.isfinite(raw_beta) and raw_beta > 0) else np.nan

    reason = "" if np.isfinite(beta) else "Fit slope not negative (unphysical result)."
    return {
        "beta_over_nu":   beta,
        "pend_amp":       float(amp) if np.isfinite(amp) else np.nan,
        "pend_fit_r2":    float(r2)  if np.isfinite(r2)  else np.nan,
        "pend_n_fit":     int(n),
        "beta_nu_reason": reason,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def first_positive_lambda(lam, pend):
    idx = np.where(np.asarray(pend) > 0)[0]
    return float(lam[idx[0]]) if len(idx) > 0 else np.nan


def crossing_lambda(lam, pend, thr):
    lam  = np.asarray(lam, float)
    pend = np.asarray(pend, float)
    idx  = np.where(pend >= thr)[0]
    if len(idx) == 0:
        return np.nan
    i = int(idx[0])
    if i == 0:
        return float(lam[0])
    p0, p1 = pend[i - 1], pend[i]
    l0, l1 = lam[i - 1],  lam[i]
    return float(l1) if p1 == p0 else float(np.interp(thr, [p0, p1], [l0, l1]))


def make_onset_diagnostics(gamma_df, peak_df, lambda_c):
    peak_map = peak_df.set_index("N")["lambda_p"].to_dict()
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)
        lam = sdf["lambda"].to_numpy(float)
        pen = sdf["P_end"].to_numpy(float)
        rows.append({
            "gamma":                float(sdf["gamma"].iloc[0]),
            "N":                    int(N),
            "lambda_p_tau":         float(peak_map.get(int(N), np.nan)),
            "lambda_c_global":      float(lambda_c) if np.isfinite(lambda_c) else np.nan,
            "lambda_first_pend_pos": first_positive_lambda(lam, pen),
            "lambda_pend_0p005":    crossing_lambda(lam, pen, 0.005),
            "lambda_pend_0p01":     crossing_lambda(lam, pen, 0.01),
            "P_end_at_lambda_c": (
                float(np.interp(lambda_c, lam, pen))
                if np.isfinite(lambda_c) and (lam[0] <= lambda_c <= lam[-1])
                else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def check_step5_feasibility(peak_df, diag_df, lambda_c, gamma):
    """
    Checks whether lambda_c falls in a region where P_end > 0 can be
    measured from the simulated data, and returns a human-readable reason
    string when beta/nu is not extractable.
    """
    lp_min = float(np.nanmin(peak_df["lambda_p"]))

    out = {
        "lp_min":                    lp_min,
        "overlap_first_pos":         np.nan,
        "overlap_0p005":             np.nan,
        "overlap_0p01":              np.nan,
        "max_lambda_first_pend_pos": np.nan,
        "max_lambda_pend_0p005":     np.nan,
        "max_lambda_pend_0p01":      np.nan,
    }

    for col, key in [
        ("lambda_first_pend_pos", "first_pos"),
        ("lambda_pend_0p005",     "0p005"),
        ("lambda_pend_0p01",      "0p01"),
    ]:
        vals = diag_df[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        vmax = float(np.max(vals))
        out[f"max_{col}"]      = vmax
        out[f"overlap_{key}"] = int(vmax < lp_min)

    # Human-readable reason for beta/nu not being extractable
    if float(gamma) < 3.0 and np.isclose(float(lambda_c), 0.0):
        out["beta_nu_feasible"] = False
        out["beta_nu_reason"]   = (
            "gamma<3: lambda_c=0 in the thermodynamic limit. "
            "P_end(0,N)=0 identically for all finite N because no epidemic "
            "can sustain itself at zero infection rate. "
            "The FSS relation P_end ~ N^{-beta/nu} at lambda_c is inapplicable."
        )
    elif np.isfinite(lambda_c) and np.isfinite(out["max_lambda_first_pend_pos"]):
        if lambda_c < out["max_lambda_first_pend_pos"]:
            out["beta_nu_feasible"] = False
            out["beta_nu_reason"]   = (
                f"lambda_c ({lambda_c:.5f}) is smaller than the smallest lambda "
                f"at which P_end>0 for the largest simulated sizes "
                f"(min onset = {out['max_lambda_first_pend_pos']:.5f}). "
                "The simulated lambda grid does not cover values between 0 and "
                "lambda_c with nonzero endemic probability. "
                "This is a simulation scan-range limitation: the Fortran code "
                "was not run at lambda values close enough to lambda_c from below."
            )
        else:
            out["beta_nu_feasible"] = True
            out["beta_nu_reason"]   = ""
    else:
        out["beta_nu_feasible"] = False
        out["beta_nu_reason"]   = "lambda_c or P_end onset not determinable."

    return out


# ---------------------------------------------------------------------------
# Per-gamma analysis
# ---------------------------------------------------------------------------

def analyze_gamma(gamma, gamma_df):
    peak_df = extract_tau_peaks(gamma_df)
    peak_df.to_csv(
        os.path.join(PROC_DIR, f"part2_peaks_gamma_{gamma:.1f}.csv"), index=False
    )

    out = {"gamma": float(gamma), "n_sizes": int(peak_df["N"].nunique())}
    out.update(estimate_lambda_c(gamma, peak_df))
    out.update(fit_tau_peak(peak_df))

    lc = out["lambda_c"]

    if np.isfinite(lc):
        pend_df = extract_pend_at_lc(gamma_df, gamma, lc)

        if float(gamma) < 3.0 and np.isclose(lc, 0.0):
            # lambda_c=0: P_end(0,N)=0 by definition — skip fit entirely
            out.update({
                "beta_over_nu":   np.nan,
                "pend_amp":       np.nan,
                "pend_fit_r2":    np.nan,
                "pend_n_fit":     0,
                "beta_nu_reason": (
                    "lambda_c=0: P_end(0,N)=0 identically. "
                    "beta/nu not extractable by construction."
                ),
            })
        else:
            out.update(fit_beta_over_nu(pend_df))
    else:
        pend_df = pd.DataFrame(
            columns=["gamma", "N", "lambda_c", "lambda_eval", "P_end_lc", "eval_mode"]
        )
        out.update({
            "beta_over_nu":   np.nan,
            "pend_amp":       np.nan,
            "pend_fit_r2":    np.nan,
            "pend_n_fit":     0,
            "beta_nu_reason": "lambda_c not determined.",
        })

    pend_df.to_csv(
        os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv"), index=False
    )

    diag_df = make_onset_diagnostics(gamma_df, peak_df, lc)
    diag_df.to_csv(
        os.path.join(PROC_DIR, f"part2_diagnostics_gamma_{gamma:.1f}.csv"), index=False
    )

    feas = check_step5_feasibility(peak_df, diag_df, lc, gamma)
    out.update({
        "lp_min":                    feas["lp_min"],
        "step5_overlap_first_pos":   feas["overlap_first_pos"],
        "step5_overlap_0p005":       feas["overlap_0p005"],
        "step5_overlap_0p01":        feas["overlap_0p01"],
        "max_lambda_first_pend_pos": feas["max_lambda_first_pend_pos"],
        "max_lambda_pend_0p005":     feas["max_lambda_pend_0p005"],
        "max_lambda_pend_0p01":      feas["max_lambda_pend_0p01"],
        "beta_nu_feasible":          feas.get("beta_nu_feasible", np.nan),
        # beta_nu_reason: prefer the one from fit_beta_over_nu if already set
        "beta_nu_reason":            out.get("beta_nu_reason") or feas.get("beta_nu_reason", ""),
    })

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_raw_results()

    print("\nFiles loaded per (gamma, N):")
    for (gamma, N), sub in data.groupby(["gamma", "N"]):
        sources = list(sub["source"].unique())
        lmin, lmax, npts = sub["lambda"].min(), sub["lambda"].max(), len(sub)
        print(
            f"  gamma={gamma}  N={N:>8d}  lambda=[{lmin:.4f},{lmax:.4f}]"
            f"  npts={npts}  sources={sources}"
        )

    summaries = []
    for gamma, gdf in data.groupby("gamma"):
        summaries.append(
            analyze_gamma(gamma, gdf.sort_values(["N", "lambda"]).reset_index(drop=True))
        )

    summary_df = pd.DataFrame(summaries).sort_values("gamma").reset_index(drop=True)
    summary_df.to_csv(os.path.join(PROC_DIR, "part2_critical_exponents.csv"), index=False)

    print("\nCritical exponents:\n")
    cols = ["gamma", "lambda_c", "inv_nu", "lambdap_fit_r2", "n_fit", "n_outliers_removed",
            "gamma1_over_nu", "tau_peak_fit_r2", "tau_peak_n_removed",
            "beta_over_nu", "pend_fit_r2", "beta_nu_feasible"]
    print(summary_df[[c for c in cols if c in summary_df.columns]].to_string(index=False))
    print()

    print("\nbeta/nu reasons:")
    for _, row in summary_df.iterrows():
        print(f"  gamma={row['gamma']}: {row.get('beta_nu_reason', '')}")

    print("\nWritten to", PROC_DIR)

    print("\nStep-5 diagnostics (P_end at lambda_c):")
    for gamma in summary_df["gamma"]:
        pf = os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv")
        if os.path.exists(pf):
            pdf = pd.read_csv(pf)
            if len(pdf) > 0:
                print(f"\n  gamma={gamma}:")
                print(pdf[["N", "lambda_c", "P_end_lc", "eval_mode"]].to_string(index=False))

    print("\nOnset diagnostics:")
    for gamma in summary_df["gamma"]:
        pf = os.path.join(PROC_DIR, f"part2_diagnostics_gamma_{gamma:.1f}.csv")
        if os.path.exists(pf):
            pdf = pd.read_csv(pf)
            if len(pdf) > 0:
                print(f"\n  gamma={gamma}:")
                print(pdf.to_string(index=False))


if __name__ == "__main__":
    main()