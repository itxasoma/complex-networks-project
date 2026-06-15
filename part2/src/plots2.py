"""
plots2.py: Plotting only for Part 2 (SIS lifespan method).

Reads:
  part2/results/raw/part2_N*_g*.dat
  part2/results/raw/part2_N*_g*_window.dat
  part2/results/processed/part2_peaks_gamma_*.csv
  part2/results/processed/part2_pend_lc_gamma_*.csv
  part2/results/processed/part2_critical_exponents.csv

Writes:
  part2/figures/part2_tau_gamma_*.pdf
  part2/figures/part2_lambdap_gamma_*.pdf
  part2/figures/part2_taupeak_gamma_*.pdf
  part2/figures/part2_pend_lc_gamma_*.pdf
  part2/figures/part2_collapse_tau_gamma_*.pdf
  part2/figures/part2_collapse_pend_gamma_*.pdf
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "part2", "results", "raw")
PROC_DIR = os.path.join(REPO_ROOT, "part2", "results", "processed")
FIG_DIR = os.path.join(REPO_ROOT, "part2", "figures")
STYLE_FILE = os.path.join(REPO_ROOT, "part1", "src", "mplstyle", "science.mplstyle")

os.makedirs(FIG_DIR, exist_ok=True)
if os.path.exists(STYLE_FILE):
    plt.style.use(STYLE_FILE)

FILE_RE = re.compile(r"^part2_N(?P<N>\d+)_g(?P<gamma>\d+(?:\.\d+)?)(?P<window>_window)?\.dat$")


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
    return data.sort_values(["gamma", "N", "lambda"]).reset_index(drop=True)


def safe_loglog(ax, x, y, *args, **kwargs):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(mask) >= 2:
        ax.loglog(x[mask], y[mask], *args, **kwargs)
        return True
    return False


def make_tau_plot(gamma, gamma_df):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for N, sdf in sorted(gamma_df.groupby("N"), key=lambda z: z[0]):
        ax.plot(sdf["lambda"], sdf["tau"], marker="o", ms=3.2, lw=1.2, label=rf"$N={N}$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle \tau(\lambda, N) \rangle$")
    ax.set_title(rf"$\gamma = {gamma:.1f}$")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"part2_tau_gamma_{gamma:.1f}.pdf"))
    plt.savefig(os.path.join(FIG_DIR, f"part2_tau_gamma_{gamma:.1f}.png"), dpi=300)
    plt.close()


def make_lambdap_plot(gamma, peak_df, summary_row):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    N = peak_df["N"].to_numpy(dtype=float)
    lp = peak_df["lambda_p"].to_numpy(dtype=float)
    lambda_c = float(summary_row["lambda_c"]) if pd.notna(summary_row["lambda_c"]) else np.nan
    A = float(summary_row["A_lambdap"]) if pd.notna(summary_row["A_lambdap"]) else np.nan
    inv_nu = float(summary_row["inv_nu"]) if pd.notna(summary_row["inv_nu"]) else np.nan
    fit_r2 = float(summary_row["lambdap_fit_r2"]) if pd.notna(summary_row["lambdap_fit_r2"]) else np.nan

    ax = axes[0]
    ax.plot(N, lp, "o", ms=4, label=r"$\lambda_p(N)$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\lambda_p(N)$")
    ax.set_title(rf"$\gamma = {gamma:.1f}$")
    if np.isfinite(lambda_c) and np.isfinite(inv_nu):
        xfit = np.logspace(np.log10(np.min(N)), np.log10(np.max(N)), 300)
        yfit = lambda_c + A * xfit ** (-inv_nu)
        ax.plot(xfit, yfit, "-", lw=1.4, label=rf"fit, $\lambda_c={lambda_c:.5f}$")
        ax.axhline(lambda_c, color="black", lw=1.0, ls="--")
    ax.legend(fontsize=7)

    ax = axes[1]
    if np.isfinite(lambda_c):
        diff = lp - lambda_c
        ok = safe_loglog(ax, N, diff, "o", ms=4, label=r"$\lambda_p-\lambda_c$")
        if ok and np.isfinite(inv_nu):
            xfit = np.logspace(np.log10(np.min(N)), np.log10(np.max(N)), 300)
            yfit = A * xfit ** (-inv_nu)
            safe_loglog(ax, xfit, yfit, "-", lw=1.4, label=rf"$1/\nu={inv_nu:.3f}$, $R^2={fit_r2:.4f}$")
            ax.legend(fontsize=7)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$|\lambda_p(N)-\lambda_c|$")
    ax.set_title("Notes-style fit")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"part2_lambdap_gamma_{gamma:.1f}.pdf"))
    plt.savefig(os.path.join(FIG_DIR, f"part2_lambdap_gamma_{gamma:.1f}.png"), dpi=300)
    plt.close()


def make_taupeak_plot(gamma, peak_df, summary_row):
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    N = peak_df["N"].to_numpy(dtype=float)
    y = peak_df["tau_peak"].to_numpy(dtype=float)
    safe_loglog(ax, N, y, "o", ms=4, label="data")

    exponent = float(summary_row["gamma1_over_nu"]) if pd.notna(summary_row["gamma1_over_nu"]) else np.nan
    amp = float(summary_row["tau_peak_amp"]) if pd.notna(summary_row["tau_peak_amp"]) else np.nan
    r2 = float(summary_row["tau_peak_fit_r2"]) if pd.notna(summary_row["tau_peak_fit_r2"]) else np.nan
    if np.isfinite(exponent) and np.isfinite(amp):
        xfit = np.logspace(np.log10(np.min(N)), np.log10(np.max(N)), 300)
        yfit = amp * xfit ** (exponent)
        safe_loglog(ax, xfit, yfit, "-", lw=1.4, label=rf"$\gamma_1/\nu={exponent:.3f}$, $R^2={r2:.4f}$")
        ax.legend(fontsize=7)

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\langle \tau \rangle_{\mathrm{peak}}$")
    ax.set_title(rf"$\gamma = {gamma:.1f}$")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"part2_taupeak_gamma_{gamma:.1f}.pdf"))
    plt.savefig(os.path.join(FIG_DIR, f"part2_taupeak_gamma_{gamma:.1f}.png"), dpi=300)
    plt.close()


def make_pend_lc_plot(gamma, pend_df, summary_row):
    if len(pend_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    N = pend_df["N"].to_numpy(dtype=float)
    y = pend_df["P_end_lc"].to_numpy(dtype=float)
    safe_loglog(ax, N, y, "o", ms=4, label="data")

    exponent = float(summary_row["beta_over_nu"]) if pd.notna(summary_row["beta_over_nu"]) else np.nan
    amp = float(summary_row["pend_amp"]) if pd.notna(summary_row["pend_amp"]) else np.nan
    r2 = float(summary_row["pend_fit_r2"]) if pd.notna(summary_row["pend_fit_r2"]) else np.nan
    if np.isfinite(exponent) and np.isfinite(amp):
        xfit = np.logspace(np.log10(np.min(N)), np.log10(np.max(N)), 300)
        yfit = amp * xfit ** (-exponent)
        safe_loglog(ax, xfit, yfit, "-", lw=1.4, label=rf"$\beta/\nu={exponent:.3f}$, $R^2={r2:.4f}$")
        ax.legend(fontsize=7)

    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$P_{\mathrm{end}}(\lambda_c, N)$")
    ax.set_title(rf"$\gamma = {gamma:.1f}$")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.pdf"))
    plt.savefig(os.path.join(FIG_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.png"), dpi=300)
    plt.close()


def make_collapse_tau_plot(gamma, gamma_df, summary_row):
    lambda_c = float(summary_row["lambda_c"]) if pd.notna(summary_row["lambda_c"]) else np.nan
    inv_nu = float(summary_row["inv_nu"]) if pd.notna(summary_row["inv_nu"]) else np.nan
    gamma1_over_nu = float(summary_row["gamma1_over_nu"]) if pd.notna(summary_row["gamma1_over_nu"]) else np.nan
    if not (np.isfinite(lambda_c) and np.isfinite(inv_nu) and np.isfinite(gamma1_over_nu)):
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for N, sdf in sorted(gamma_df.groupby("N"), key=lambda z: z[0]):
        x = (sdf["lambda"].to_numpy(dtype=float) - lambda_c) * (float(N) ** inv_nu)
        y = sdf["tau"].to_numpy(dtype=float) / (float(N) ** gamma1_over_nu)
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(mask) >= 2:
            ax.plot(x[mask], y[mask], marker="o", ms=3.0, lw=1.0, label=rf"$N={N}$")
    ax.set_xlabel(r"$(\lambda - \lambda_c) N^{1/\nu}$")
    ax.set_ylabel(r"$\langle \tau \rangle / N^{\gamma_1/\nu}$")
    ax.set_title(rf"$\gamma = {gamma:.1f}$")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"part2_collapse_tau_gamma_{gamma:.1f}.pdf"))
    plt.savefig(os.path.join(FIG_DIR, f"part2_collapse_tau_gamma_{gamma:.1f}.png"), dpi=300)
    plt.close()


def make_collapse_pend_plot(gamma, gamma_df, summary_row):
    lambda_c = float(summary_row["lambda_c"]) if pd.notna(summary_row["lambda_c"]) else np.nan
    inv_nu = float(summary_row["inv_nu"]) if pd.notna(summary_row["inv_nu"]) else np.nan
    beta_over_nu = float(summary_row["beta_over_nu"]) if pd.notna(summary_row["beta_over_nu"]) else np.nan
    if not (np.isfinite(lambda_c) and np.isfinite(inv_nu) and np.isfinite(beta_over_nu)):
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for N, sdf in sorted(gamma_df.groupby("N"), key=lambda z: z[0]):
        x = (sdf["lambda"].to_numpy(dtype=float) - lambda_c) * (float(N) ** inv_nu)
        y = sdf["P_end"].to_numpy(dtype=float) * (float(N) ** beta_over_nu)
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(mask) >= 2:
            ax.plot(x[mask], y[mask], marker="o", ms=3.0, lw=1.0, label=rf"$N={N}$")
    ax.set_xlabel(r"$(\lambda - \lambda_c) N^{1/\nu}$")
    ax.set_ylabel(r"$P_{\mathrm{end}} N^{\beta/\nu}$")
    ax.set_title(rf"$\gamma = {gamma:.1f}$")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"part2_collapse_pend_gamma_{gamma:.1f}.pdf"))
    plt.savefig(os.path.join(FIG_DIR, f"part2_collapse_pend_gamma_{gamma:.1f}.png"), dpi=300)
    plt.close()


def main():
    data = load_raw_results()
    summary = pd.read_csv(os.path.join(PROC_DIR, "part2_critical_exponents.csv"))

    for gamma, gamma_df in data.groupby("gamma"):
        gamma_df = gamma_df.sort_values(["N", "lambda"]).reset_index(drop=True)
        summary_row = summary.loc[summary["gamma"] == gamma].iloc[0]
        peak_df = pd.read_csv(os.path.join(PROC_DIR, f"part2_peaks_gamma_{gamma:.1f}.csv"))
        pend_path = os.path.join(PROC_DIR, f"part2_pend_lc_gamma_{gamma:.1f}.csv")
        pend_df = pd.read_csv(pend_path) if os.path.exists(pend_path) else pd.DataFrame()

        make_tau_plot(gamma, gamma_df)
        make_lambdap_plot(gamma, peak_df, summary_row)
        make_taupeak_plot(gamma, peak_df, summary_row)
        make_pend_lc_plot(gamma, pend_df, summary_row)
        make_collapse_tau_plot(gamma, gamma_df, summary_row)
        make_collapse_pend_plot(gamma, gamma_df, summary_row)

    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
