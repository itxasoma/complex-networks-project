# ============================================================
# scaling2.py  –  Strict assignment analysis + diagnostics
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

MIN_SIZES_FOR_LC = 4
MIN_POINTS_FIT   = 3
TAIL_FRACTION    = 0.85
INV_NU_MIN, INV_NU_MAX = 0.05, 5.0

EXCLUDE_N = {2.5: {1_000_000}}

FILE_RE = re.compile(
    r"^part2_N(?P<N>\d+)_g(?P<gamma>\d+(?:\.\d+)?)(?P<window>_window\d*)?\.dat$"
)

# IO

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
    df["source"] = name
    df["is_window"] = bool(m.group("window"))
    df["source_priority"] = np.where(df["is_window"], 1, 0)
    return df

def merge_case_scans(case_df):
    case_df = case_df.sort_values(["lambda", "source_priority"]).reset_index(drop=True)

    if not case_df["is_window"].any():
        return (
            case_df
            .drop_duplicates(subset=["lambda"], keep="last")
            .sort_values("lambda")
            .reset_index(drop=True)
        )

    coarse = case_df.loc[~case_df["is_window"]].copy()
    window = case_df.loc[case_df["is_window"]].copy()

    lo, hi = float(window["lambda"].min()), float(window["lambda"].max())
    coarse = coarse.loc[(coarse["lambda"] < lo) | (coarse["lambda"] > hi)].copy()

    merged = pd.concat([coarse, window], ignore_index=True)
    merged = (
        merged
        .sort_values(["lambda", "source_priority"])
        .drop_duplicates(subset=["lambda"], keep="last")
        .sort_values("lambda")
        .reset_index(drop=True)
    )
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

# basic fits

def loglog_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)

    lx, ly = np.log(x), np.log(y)
    s, ic = np.polyfit(lx, ly, 1)

    yhat = s * lx + ic
    ss_res = np.sum((ly - yhat) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return float(s), float(np.exp(ic)), float(r2), len(x)

def choose_tail(df):
    df = df.sort_values("N").reset_index(drop=True)
    keep = max(MIN_POINTS_FIT, int(np.ceil(TAIL_FRACTION * len(df))))
    return df.iloc[len(df) - keep:].copy()

# Step 3: tau peaks

def local_quadratic_peak(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    i = int(np.argmax(y))

    if i == 0 or i == len(x) - 1:
        return float(x[i]), float(y[i]), i, False

    xx, yy = x[i-1:i+2], y[i-1:i+2]
    try:
        a, b, c = np.polyfit(xx, yy, 2)
        if a < 0:
            xv = -b / (2 * a)
            if xx[0] <= xv <= xx[-1]:
                yv = a * xv**2 + b * xv + c
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
            "gamma": float(sdf["gamma"].iloc[0]),
            "N": int(N),
            "lambda_p": lp,
            "tau_peak": tp,
            "lambda_argmax": float(lam[imax]),
            "tau_argmax": float(tau[imax]),
            "peak_refined": int(refined),
        })

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

def estimate_lambda_c(gamma, peak_df):
    if float(gamma) < 3.0:
        fit_df = choose_tail(peak_df)
        N  = fit_df["N"].to_numpy(float)
        lp = fit_df["lambda_p"].to_numpy(float)

        s, amp, r2, n_fit = loglog_fit(N, lp)
        inv_nu = -s if np.isfinite(s) else np.nan

        if not np.isfinite(inv_nu) or inv_nu < INV_NU_MIN or inv_nu > INV_NU_MAX:
            inv_nu, amp, r2 = np.nan, np.nan, np.nan

        return {
            "lambda_c": 0.0,
            "A_lambdap": float(amp) if np.isfinite(amp) else np.nan,
            "inv_nu": float(inv_nu) if np.isfinite(inv_nu) else np.nan,
            "lambdap_fit_r2": float(r2) if np.isfinite(r2) else np.nan,
            "n_fit": int(n_fit),
        }

    if len(peak_df) < MIN_SIZES_FOR_LC:
        return {
            "lambda_c": np.nan,
            "A_lambdap": np.nan,
            "inv_nu": np.nan,
            "lambdap_fit_r2": np.nan,
            "n_fit": 0,
        }

    fit_df = choose_tail(peak_df).sort_values("N").reset_index(drop=True)
    N  = fit_df["N"].to_numpy(float)
    lp = fit_df["lambda_p"].to_numpy(float)

    lp_min = float(np.min(lp))
    lp_max = float(np.max(lp))
    span = max(lp_max - lp_min, 1e-6)

    # Prevent the optimizer from collapsing to an unphysical lambda_c ≈ 0
    # just because that gives a numerically cleaner straight line.
    local_band = max(0.01, 0.5 * span)
    lo = max(1e-6, lp_min - local_band)
    hi = lp_min - 1e-6

    best = None
    for lc in np.linspace(lo, hi, 4000):
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
            "lambda_c": float(lc),
            "A_lambdap": float(amp),
            "inv_nu": float(inv_nu),
            "lambdap_fit_r2": float(r2),
            "n_fit": int(n_used),
        }

        if best is None:
            best = cand
        else:
            better_r2 = cand["lambdap_fit_r2"] > best["lambdap_fit_r2"] + 1e-12
            tie_r2 = abs(cand["lambdap_fit_r2"] - best["lambdap_fit_r2"]) <= 1e-12
            larger_lc = cand["lambda_c"] > best["lambda_c"]
            if better_r2 or (tie_r2 and larger_lc):
                best = cand

    if best is None:
        return {
            "lambda_c": np.nan,
            "A_lambdap": np.nan,
            "inv_nu": np.nan,
            "lambdap_fit_r2": np.nan,
            "n_fit": 0,
        }

    return best

# Step 4: tau peak height

def fit_tau_peak(peak_df):
    fit_df = choose_tail(peak_df)
    s, amp, r2, n = loglog_fit(
        fit_df["N"].to_numpy(float),
        fit_df["tau_peak"].to_numpy(float)
    )
    return {
        "gamma1_over_nu": float(s) if np.isfinite(s) else np.nan,
        "tau_peak_amp": float(amp) if np.isfinite(amp) else np.nan,
        "tau_peak_fit_r2": float(r2) if np.isfinite(r2) else np.nan,
        "tau_peak_n_fit": int(n),
    }

# Step 5: P_end(lambda_c, N)

def extract_pend_at_lc(gamma_df, gamma, lambda_c):
    rows = []
    for N, sdf in gamma_df.groupby("N"):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)

        lam  = sdf["lambda"].to_numpy(float)
        pend = sdf["P_end"].to_numpy(float)

        if lambda_c < lam[0]:
            p_val = np.nan
            mode = "out_of_range_low"
        elif lambda_c > lam[-1]:
            p_val = np.nan
            mode = "out_of_range_high"
        else:
            p_val = float(np.interp(lambda_c, lam, pend))
            mode = "interp" if p_val > 0 else "interp_zero"

        rows.append({
            "gamma": float(gamma),
            "N": int(N),
            "lambda_c": float(lambda_c),
            "lambda_eval": float(lambda_c),
            "P_end_lc": p_val if (np.isfinite(p_val) and p_val > 0) else np.nan,
            "eval_mode": mode,
        })

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

def fit_beta_over_nu(pend_df):
    fit_df = choose_tail(pend_df)
    valid = fit_df.dropna(subset=["P_end_lc"]).copy()
    valid = valid.loc[valid["P_end_lc"] > 0]

    s, amp, r2, n = loglog_fit(
        valid["N"].to_numpy(float),
        valid["P_end_lc"].to_numpy(float)
    )

    raw_beta = -s if np.isfinite(s) else np.nan
    beta = raw_beta if (np.isfinite(raw_beta) and raw_beta > 0) else np.nan

    return {
        "beta_over_nu": beta,
        "pend_amp": float(amp) if np.isfinite(amp) else np.nan,
        "pend_fit_r2": float(r2) if np.isfinite(r2) else np.nan,
        "pend_n_fit": int(n),
    }

# Diagnostics

def first_positive_lambda(lam, pend):
    idx = np.where(np.asarray(pend) > 0)[0]
    if len(idx) == 0:
        return np.nan
    return float(lam[idx[0]])

def crossing_lambda(lam, pend, thr):
    lam = np.asarray(lam, float)
    pend = np.asarray(pend, float)

    idx = np.where(pend >= thr)[0]
    if len(idx) == 0:
        return np.nan

    i = int(idx[0])
    if i == 0:
        return float(lam[0])

    p0, p1 = pend[i-1], pend[i]
    l0, l1 = lam[i-1], lam[i]

    if p1 == p0:
        return float(l1)

    return float(np.interp(thr, [p0, p1], [l0, l1]))

def make_onset_diagnostics(gamma_df, peak_df, lambda_c):
    peak_map = peak_df.set_index("N")["lambda_p"].to_dict()
    rows = []

    for N, sdf in gamma_df.groupby("N"):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)

        lam = sdf["lambda"].to_numpy(float)
        pen = sdf["P_end"].to_numpy(float)

        rows.append({
            "gamma": float(sdf["gamma"].iloc[0]),
            "N": int(N),
            "lambda_p_tau": float(peak_map.get(int(N), np.nan)),
            "lambda_c_global": float(lambda_c) if np.isfinite(lambda_c) else np.nan,
            "lambda_first_pend_pos": first_positive_lambda(lam, pen),
            "lambda_pend_0p005": crossing_lambda(lam, pen, 0.005),
            "lambda_pend_0p01": crossing_lambda(lam, pen, 0.01),
            "P_end_at_lambda_c": (
                float(np.interp(lambda_c, lam, pen))
                if np.isfinite(lambda_c) and (lam[0] <= lambda_c <= lam[-1])
                else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

def check_step5_feasibility(peak_df, diag_df):
    """
    Step 3 demands lambda_c < min(lambda_p_tau).
    Step 5 needs lambda_c where P_end(lambda_c,N) is nonzero/measurable.
    This checks whether those requirements overlap at all.
    """
    lp_min = float(np.nanmin(peak_df["lambda_p"]))

    out = {
        "lp_min": lp_min,
        "overlap_first_pos": np.nan,
        "overlap_0p005": np.nan,
        "overlap_0p01": np.nan,
        "max_lambda_first_pend_pos": np.nan,
        "max_lambda_pend_0p005": np.nan,
        "max_lambda_pend_0p01": np.nan,
    }

    for col, key in [
        ("lambda_first_pend_pos", "first_pos"),
        ("lambda_pend_0p005", "0p005"),
        ("lambda_pend_0p01", "0p01"),
    ]:
        vals = diag_df[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue

        vmax = float(np.max(vals))
        out[f"max_{col}"] = vmax
        out[f"overlap_{key}"] = int(vmax < lp_min)

    return out

# analysis for every gamma

def analyze_gamma(gamma, gamma_df):
    peak_df = extract_tau_peaks(gamma_df)
    peak_df.to_csv(
        os.path.join(PROC_DIR, f"part2_peaks_gamma_{gamma:.1f}.csv"),
        index=False
    )

    out = {"gamma": float(gamma), "n_sizes": int(peak_df["N"].nunique())}
    out.update(estimate_lambda_c(gamma, peak_df))
    out.update(fit_tau_peak(peak_df))

    lc = out["lambda_c"]

    if np.isfinite(lc):
        pend_df = extract_pend_at_lc(gamma_df, gamma, lc)

        if float(gamma) < 3.0 and lc == 0.0:
            out.update({
                "beta_over_nu": np.nan,
                "pend_amp": np.nan,
                "pend_fit_r2": np.nan,
                "pend_n_fit": 0,
            })
        else:
            out.update(fit_beta_over_nu(pend_df))
    else:
        pend_df = pd.DataFrame(
            columns=["gamma", "N", "lambda_c", "lambda_eval", "P_end_lc", "eval_mode"]
        )
        out.update({
            "beta_over_nu": np.nan,
            "pend_amp": np.nan,
            "pend_fit_r2": np.nan,
            "pend_n_fit": 0,
        })

    pend_df.to_csv(
        os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv"),
        index=False
    )

    diag_df = make_onset_diagnostics(gamma_df, peak_df, lc)
    diag_df.to_csv(
        os.path.join(PROC_DIR, f"part2_diagnostics_gamma_{gamma:.1f}.csv"),
        index=False
    )

    feas = check_step5_feasibility(peak_df, diag_df)
    out["lp_min"] = feas["lp_min"]
    out["step5_overlap_first_pos"] = feas["overlap_first_pos"]
    out["step5_overlap_0p005"] = feas["overlap_0p005"]
    out["step5_overlap_0p01"] = feas["overlap_0p01"]
    out["max_lambda_first_pend_pos"] = feas["max_lambda_first_pend_pos"]
    out["max_lambda_pend_0p005"] = feas["max_lambda_pend_0p005"]
    out["max_lambda_pend_0p01"] = feas["max_lambda_pend_0p01"]

    return out

# main

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
    print(summary_df.to_string(index=False))
    print("\nWritten to", PROC_DIR)

    print("\nAssignment step 5 diagnostics:")
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