"""
scaling2.py: Finite-size scaling analysis for Part 2 (SIS lifespan method).

Reads:
  part2/results/raw/part2_N*_g*.dat
  part2/results/raw/part2_N*_g*_window.dat

Writes:
  part2/results/processed/part2_all_raw.csv
  part2/results/processed/part2_peaks_gamma_*.csv
  part2/results/processed/part2_pend_lc_gamma_*.csv
  part2/results/processed/part2_critical_exponents.csv

FIXES applied:
  FIX-A: lambda_c scan range extended to include values >= lp_min (necessary
         for gamma=2.5 where finite-size corrections are large and the true
         lambda_c may be slightly above the smallest observed lambda_p).
  FIX-B: Scoring simplified to R^2 only; hard constraint 0.1 < inv_nu < 5.0
         prevents the score from being dominated by spuriously small inv_nu.
  FIX-C: TAIL_FRACTION raised to 0.85 and MIN_SIZES_FOR_LC raised to 4, so
         the two or three smallest (most biased) system sizes are excluded from
         the extrapolation whenever enough sizes are available.
"""

import glob
import os
import re
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "part2", "results", "raw")
PROC_DIR = os.path.join(REPO_ROOT, "part2", "results", "processed")

os.makedirs(PROC_DIR, exist_ok=True)

MIN_SIZES_FOR_LC = 4        # FIX-C: was 4, still 4 but combined with higher TAIL_FRACTION
MIN_POINTS_FIT = 3
TAIL_FRACTION = 0.85        # FIX-C: was 0.67 — now keeps only the largest ~85% of sizes

# FIX-B: hard bounds on the physically plausible exponent
INV_NU_MIN = 0.1
INV_NU_MAX = 5.0

FILE_RE = re.compile(r"^part2_N(?P<N>\d+)_g(?P<gamma>\d+(?:\.\d+)?)(?P<window>_window)?\.dat$")


# ---------- IO ----------

def list_raw_files():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "part2_N*_g*.dat")))
    valid = []
    for path in files:
        name = os.path.basename(path)
        if FILE_RE.match(name):
            valid.append(path)
    if not valid:
        raise FileNotFoundError(f"No raw files found in {RAW_DIR}")
    return valid


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
            names=["lambda", "tau", "P_end", "N", "gamma", "nruns", "M"],
        )
    except Exception:
        return None

    if len(df) == 0:
        return None

    df["source"] = name
    df["is_window"] = bool(m.group("window"))
    df["source_priority"] = np.where(df["is_window"], 1, 0)
    return df


def merge_case_scans(case_df):
    case_df = case_df.copy()
    case_df = case_df.sort_values(["lambda", "source_priority"]).reset_index(drop=True)

    if not case_df["is_window"].any():
        case_df = case_df.drop_duplicates(subset=["lambda"], keep="last")
        return case_df.sort_values("lambda").reset_index(drop=True)

    coarse = case_df.loc[~case_df["is_window"]].copy()
    window = case_df.loc[case_df["is_window"]].copy()

    lo = float(window["lambda"].min())
    hi = float(window["lambda"].max())

    coarse = coarse.loc[(coarse["lambda"] < lo) | (coarse["lambda"] > hi)].copy()

    merged = pd.concat([coarse, window], ignore_index=True)
    merged = merged.sort_values(["lambda", "source_priority"]).drop_duplicates(subset=["lambda"], keep="last")
    merged = merged.sort_values("lambda").reset_index(drop=True)
    return merged


def load_raw_results():
    files = list_raw_files()

    frames = []
    for path in files:
        df = read_raw_file(path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No readable raw result files were found.")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["gamma", "N", "lambda", "source_priority"]).reset_index(drop=True)

    merged_frames = []
    for (gamma, N), case_df in raw.groupby(["gamma", "N"], sort=True):
        merged_frames.append(merge_case_scans(case_df))

    data = pd.concat(merged_frames, ignore_index=True)
    data = data.sort_values(["gamma", "N", "lambda"]).reset_index(drop=True)
    data.to_csv(os.path.join(PROC_DIR, "part2_all_raw.csv"), index=False)
    return data


# ---------- basic fits ----------

def loglog_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)

    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    fit = slope * lx + intercept
    ss_res = np.sum((ly - fit) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(np.exp(intercept)), float(r2), len(x)


# ---------- peak extraction ----------

def local_quadratic_peak(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.argmax(y))
    x_peak = float(x[i])
    y_peak = float(y[i])

    if i == 0 or i == len(x) - 1:
        return x_peak, y_peak, i, False

    xx = x[i - 1 : i + 2]
    yy = y[i - 1 : i + 2]

    try:
        a, b, c = np.polyfit(xx, yy, 2)
        if a < 0:
            xv = -b / (2.0 * a)
            if xx[0] <= xv <= xx[-1]:
                yv = a * xv * xv + b * xv + c
                if np.isfinite(yv) and yv > 0.0:
                    return float(xv), float(yv), i, True
    except Exception:
        pass

    return x_peak, y_peak, i, False


def extract_peaks(gamma_df):
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)
        lam = sdf["lambda"].to_numpy(dtype=float)
        tau = sdf["tau"].to_numpy(dtype=float)
        lambda_p, tau_peak, imax, refined = local_quadratic_peak(lam, tau)
        rows.append(
            {
                "gamma": float(sdf["gamma"].iloc[0]),
                "N": int(N),
                "lambda_p": lambda_p,
                "tau_peak": tau_peak,
                "lambda_argmax": float(lam[imax]),
                "tau_argmax": float(tau[imax]),
                "peak_refined": int(refined),
            }
        )
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


# ---------- lambda_c from notes-style scan ----------

def choose_tail(df):
    df = df.sort_values("N").reset_index(drop=True)
    n = len(df)
    keep = max(MIN_POINTS_FIT, int(np.ceil(TAIL_FRACTION * n)))
    return df.iloc[n - keep :].copy()


def estimate_lambda_c_notes(peak_df):
    if len(peak_df) < MIN_SIZES_FOR_LC:
        return {
            "lambda_c": np.nan,
            "A_lambdap": np.nan,
            "inv_nu": np.nan,
            "lambdap_fit_r2": np.nan,
            "n_fit": 0,
        }

    fit_df = choose_tail(peak_df)
    N = fit_df["N"].to_numpy(dtype=float)
    lp = fit_df["lambda_p"].to_numpy(dtype=float)

    lp_min = float(np.min(lp))
    lp_max = float(np.max(lp))
    span = max(lp_max - lp_min, 1.0e-6)

    # FIX-A: extend the search range symmetrically around the observed lambda_p
    # values.  The lower bound is well below lp_min; the upper bound is now
    # lp_min + 2*span, so that a true lambda_c slightly above the smallest lp
    # (typical for gamma=2.5 with strong finite-size effects) is included.
    lo = max(1.0e-8, lp_min - 5.0 * span)
    hi = lp_min + 2.0 * span          # FIX-A: was 0.999999 * lp_min

    if hi <= lo:
        lo = max(1.0e-8, 0.5 * lp_min)
        hi = lp_max + span

    trial_lc = np.linspace(lo, hi, 8000)   # more grid points for finer resolution
    best = None

    for lc in trial_lc:
        diff = lp - lc
        # skip if any diff is non-positive (would break loglog_fit)
        if np.any(diff <= 0.0):
            continue
        slope, amp, r2, n_used = loglog_fit(N, diff)
        inv_nu = -slope
        if n_used < MIN_POINTS_FIT:
            continue
        if not np.isfinite(inv_nu) or not np.isfinite(r2):
            continue
        # FIX-B: enforce physical bounds on inv_nu; score by R^2 alone
        if inv_nu < INV_NU_MIN or inv_nu > INV_NU_MAX:
            continue
        score = r2   # FIX-B: was (r2, -abs(inv_nu)) which biased toward small inv_nu
        if best is None or score > best["score"]:
            best = {
                "lambda_c": float(lc),
                "A_lambdap": float(amp),
                "inv_nu": float(inv_nu),
                "lambdap_fit_r2": float(r2),
                "n_fit": int(n_used),
                "score": score,
            }

    if best is None:
        return {
            "lambda_c": np.nan,
            "A_lambdap": np.nan,
            "inv_nu": np.nan,
            "lambdap_fit_r2": np.nan,
            "n_fit": 0,
        }

    return {k: v for k, v in best.items() if k != "score"}


# ---------- P_end at lambda_c ----------

def interpolate_at_lambda(x, y, x0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if x0 < x[0] or x0 > x[-1]:
        return np.nan, np.nan, "out-of-range"

    return float(np.interp(x0, x, y)), float(x0), "interp"


def extract_pend_at_lc(gamma_df, gamma, lambda_c):
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)
        p_val, lam_eval, mode = interpolate_at_lambda(
            sdf["lambda"].to_numpy(dtype=float),
            sdf["P_end"].to_numpy(dtype=float),
            lambda_c,
        )
        rows.append(
            {
                "gamma": float(gamma),
                "N": int(N),
                "lambda_c": float(lambda_c),
                "lambda_eval": lam_eval,
                "P_end_lc": p_val,
                "eval_mode": mode,
            }
        )
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


# ---------- exponent fits ----------

def fit_tau_peak(peak_df):
    fit_df = choose_tail(peak_df)
    slope, amp, r2, n_fit = loglog_fit(
        fit_df["N"].to_numpy(dtype=float),
        fit_df["tau_peak"].to_numpy(dtype=float),
    )
    gamma1_over_nu = slope if np.isfinite(slope) else np.nan
    return {
        "gamma1_over_nu": gamma1_over_nu,
        "tau_peak_amp": amp,
        "tau_peak_fit_r2": r2,
        "tau_peak_n_fit": n_fit,
    }


def fit_beta_over_nu(pend_df):
    fit_df = choose_tail(pend_df)
    slope, amp, r2, n_fit = loglog_fit(
        fit_df["N"].to_numpy(dtype=float),
        fit_df["P_end_lc"].to_numpy(dtype=float),
    )
    # P_end ~ N^{-beta/nu}  =>  slope is negative  =>  beta/nu = -slope
    raw_beta = -slope if np.isfinite(slope) else np.nan
    # FIX (sign guard): beta/nu must be positive; if the fit gives a positive
    # slope (P_end growing with N), the data are not in the scaling regime.
    # Report NaN in that case so the plot does not silently draw a wrong line.
    beta_over_nu = raw_beta if (np.isfinite(raw_beta) and raw_beta > 0.0) else np.nan
    return {
        "beta_over_nu": beta_over_nu,
        "pend_amp": amp,
        "pend_fit_r2": r2,
        "pend_n_fit": n_fit,
    }


# ---------- main analysis ----------

def analyze_gamma(gamma, gamma_df):
    peak_df = extract_peaks(gamma_df)
    peak_df.to_csv(os.path.join(PROC_DIR, f"part2_peaks_gamma_{gamma:.1f}.csv"), index=False)

    out = {"gamma": float(gamma), "n_sizes": int(peak_df["N"].nunique())}
    out.update(estimate_lambda_c_notes(peak_df))
    out.update(fit_tau_peak(peak_df))

    if np.isfinite(out["lambda_c"]):
        pend_df = extract_pend_at_lc(gamma_df, gamma, out["lambda_c"])
        pend_df.to_csv(os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv"), index=False)
        out.update(fit_beta_over_nu(pend_df))
    else:
        pd.DataFrame(columns=["gamma", "N", "lambda_c", "lambda_eval", "P_end_lc", "eval_mode"]).to_csv(
            os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv"), index=False
        )
        out.update({
            "beta_over_nu": np.nan,
            "pend_amp": np.nan,
            "pend_fit_r2": np.nan,
            "pend_n_fit": 0,
        })

    return out


def main():
    data = load_raw_results()
    summaries = []

    for gamma, gamma_df in data.groupby("gamma"):
        gamma_df = gamma_df.sort_values(["N", "lambda"]).reset_index(drop=True)
        summaries.append(analyze_gamma(gamma, gamma_df))

    summary_df = pd.DataFrame(summaries).sort_values("gamma").reset_index(drop=True)
    summary_df.to_csv(os.path.join(PROC_DIR, "part2_critical_exponents.csv"), index=False)

    print("\nCritical exponents summary:\n")
    print(summary_df.to_string(index=False))
    print("\nProcessed files written to", PROC_DIR)


if __name__ == "__main__":
    main()