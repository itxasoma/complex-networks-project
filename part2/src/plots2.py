# ============================================================
# plots2.py  –  Figures for Part 2 (SIS lifespan FSS)
# ============================================================
"""
Produces exactly 3 compound figures per gamma:
(1) raw tau curves, (2) FSS scaling fits, (3) data collapse.

"""

import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR    = os.path.join(REPO_ROOT, "part2", "results", "raw")
PROC_DIR   = os.path.join(REPO_ROOT, "part2", "results", "processed")
FIG_DIR    = os.path.join(REPO_ROOT, "part2", "figures")
STYLE_FILE = os.path.join(REPO_ROOT, "part1", "src", "mplstyle", "science.mplstyle")

os.makedirs(FIG_DIR, exist_ok=True)
if os.path.exists(STYLE_FILE):
    plt.style.use(STYLE_FILE)

FILE_RE    = re.compile(r"^part2_N(?P<N>\d+)_g(?P<gamma>\d+(?:\.\d+)?)(?P<window>_window)?\.dat$")
P_END_EXCL = 0.95
EXCLUDE_N  = {}   

_ALL_NS   = [10_000, 30_000, 50_000, 100_000, 300_000, 500_000, 1_000_000]
_INFERNO  = matplotlib.colormaps["inferno"]   # FIX-8: replaces deprecated get_cmap()
# sample between 0.10 and 0.85 to stay away from near-black and near-white ends
_N_COLORS = {
    N: _INFERNO(0.10 + 0.75 * i / (len(_ALL_NS) - 1))
    for i, N in enumerate(_ALL_NS)
}

def color(N): return _N_COLORS.get(int(N), "#333333")
def nlabel(N): return f"$N={N//1000}$k" if int(N) < 1_000_000 else "$N=1$M"

# ── IO (as scaling2.py) ──────────────────────────────────────

def read_raw_file(path):
    name = os.path.basename(path)
    m = FILE_RE.match(name)
    if m is None: return None
    try:
        df = pd.read_csv(path, comment="#", sep=r"\s+", header=None,
                         names=["lambda","tau","P_end","N","gamma","nruns","M"])
    except Exception: return None
    if len(df) == 0: return None
    df["source"]          = name
    df["is_window"]       = bool(m.group("window"))
    df["source_priority"] = np.where(df["is_window"], 1, 0)
    return df

def merge_case_scans(case_df):
    case_df = case_df.sort_values(["lambda","source_priority"]).reset_index(drop=True)
    if not case_df["is_window"].any():
        return (case_df.drop_duplicates(subset=["lambda"], keep="last")
                       .sort_values("lambda").reset_index(drop=True))
    coarse = case_df.loc[~case_df["is_window"]].copy()
    window = case_df.loc[case_df["is_window"]].copy()
    lo, hi = float(window["lambda"].min()), float(window["lambda"].max())
    coarse = coarse.loc[(coarse["lambda"] < lo) | (coarse["lambda"] > hi)].copy()
    merged = pd.concat([coarse, window], ignore_index=True)
    return (merged.sort_values(["lambda","source_priority"])
                  .drop_duplicates(subset=["lambda"], keep="last")
                  .sort_values("lambda").reset_index(drop=True))

def load_raw_results():
    files  = sorted(glob.glob(os.path.join(RAW_DIR, "part2_N*_g*.dat")))
    files  = [p for p in files if FILE_RE.match(os.path.basename(p))]
    frames = [read_raw_file(p) for p in files]
    frames = [f for f in frames if f is not None]
    if not frames: raise RuntimeError("No raw files found.")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["gamma","N","lambda","source_priority"]).reset_index(drop=True)
    merged = []
    for (gamma, N), sub in raw.groupby(["gamma","N"], sort=True):
        if N in EXCLUDE_N.get(float(gamma), set()):  # FIX-4
            continue
        merged.append(merge_case_scans(sub))
    data = pd.concat(merged, ignore_index=True)
    return data.sort_values(["gamma","N","lambda"]).reset_index(drop=True)

def safe_loglog(ax, x, y, *a, **kw):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() >= 2:
        ax.loglog(x[mask], y[mask], *a, **kw)
        return True
    return False

# ── Figure 1: tau(lambda) curves ─────────────────────────────────────────────

def make_tau_plot(gamma, gamma_df, peak_df):
    """FIX-4/5: N=1M excluded for gamma<3; x-axis zoomed; peaks marked."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    all_lp = []
    for N, sdf in sorted(gamma_df.groupby("N"), key=lambda z: int(z[0])):
        sdf = sdf.sort_values("lambda").reset_index(drop=True)
        excluded = (float(gamma) < 3.0 and int(N) == 1_000_000)
        ls = "--" if excluded else "-"
        lw = 1.0 if excluded else 1.5
        ax.plot(sdf["lambda"], sdf["tau"],
                color=color(N), lw=lw, ls=ls,
                label=nlabel(N) + (" (excl.)" if excluded else ""))
        row = peak_df.loc[peak_df["N"] == N]
        if len(row) and not excluded:
            lp = float(row["lambda_p"].iloc[0])
            tp = float(row["tau_peak"].iloc[0])
            ax.plot(lp, tp, "x", color=color(N), ms=6, mew=1.5)
            all_lp.append(lp)

    if all_lp:
        margin = max(0.005, 0.4 * np.ptp(all_lp))
        ax.set_xlim(max(0, min(all_lp) - margin), max(all_lp) + margin)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\langle \tau(\lambda, N) \rangle$")
    ax.set_title(rf"$\gamma = {gamma:.1f}$ mean lifespan")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.text(0.97, 0.05, r"$\times$ peak $\lambda_p(N)$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="0.4")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"part2_tau_gamma_{gamma:.1f}.{ext}"),
                    dpi=300 if ext == "png" else None)
    plt.close()

# ── Figure 2: FSS scaling (lambda_p and tau_peak) ─────────────────────────────

def make_fss_plot(gamma, peak_df, summary_row):
    """
    Two-panel figure: left = lambda_p(N), right = tau_peak(N).
    FIX-2: for gamma<3 the left panel shows a raw log-log fit (lambda_c=0,
           no subtraction); the right panel annotates that slope~0 is expected.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    N   = peak_df["N"].to_numpy(float)
    lp  = peak_df["lambda_p"].to_numpy(float)
    tp  = peak_df["tau_peak"].to_numpy(float)
    lc  = float(summary_row["lambda_c"])        if pd.notna(summary_row["lambda_c"])        else np.nan
    A   = float(summary_row["A_lambdap"])       if pd.notna(summary_row["A_lambdap"])       else np.nan
    inv = float(summary_row["inv_nu"])          if pd.notna(summary_row["inv_nu"])          else np.nan
    r2  = float(summary_row["lambdap_fit_r2"])  if pd.notna(summary_row["lambdap_fit_r2"])  else np.nan
    g1  = float(summary_row["gamma1_over_nu"])  if pd.notna(summary_row["gamma1_over_nu"])  else np.nan
    ta  = float(summary_row["tau_peak_amp"])    if pd.notna(summary_row["tau_peak_amp"])    else np.nan
    tr2 = float(summary_row["tau_peak_fit_r2"]) if pd.notna(summary_row["tau_peak_fit_r2"]) else np.nan

    # ── left panel: lambda_p(N) ──────────────────────────────────────────────
    ax = axes[0]
    for i in range(len(N)):
        ax.scatter(N[i], lp[i], color=color(int(N[i])), s=25, zorder=3)

    if np.isfinite(inv) and np.isfinite(A):
        xfit = np.geomspace(N.min() * 0.7, N.max() * 1.5, 300)
        if float(gamma) < 3.0:
            yfit = A * xfit ** (-inv)
            ax.loglog(xfit, yfit, "--", color="tab:red", lw=1.5,
                      label=rf"$\lambda_p \sim N^{{-1/\nu}}$, $1/\nu={inv:.3f}$"
                            rf", $R^2={r2:.3f}$")
            ax.axhline(0, color="black", lw=0.8, ls=":", label=r"$\lambda_c = 0$")
        else:
            diff = lp - lc
            if np.all(diff > 0):
                for i in range(len(N)):
                    ax.scatter(N[i], diff[i], color=color(int(N[i])), s=25,
                               zorder=3, marker="D")
                yfit = A * xfit ** (-inv)
                safe_loglog(ax, xfit, yfit, "--", color="tab:red", lw=1.5,
                            label=rf"$1/\nu={inv:.3f}$, $R^2={r2:.3f}$")
                ax.axhline(lc, color="black", lw=0.8, ls="--",
                           label=rf"$\lambda_c = {lc:.4f}$")
            else:
                ax.plot(N, diff, "o", color="tab:orange", ms=5)
                ax.axhline(0, color="black", lw=0.8, ls="--")
                ax.set_xscale("log")
                ax.text(0.05, 0.92, "WARNING: λ_c overestimated\nfor some sizes",
                        transform=ax.transAxes, fontsize=7,
                        va="top", color="tab:red")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\lambda_p(N)$" if float(gamma) < 3.0
                  else r"$\lambda_p(N) - \lambda_c$")
    ax.set_title(rf"$\gamma={gamma:.1f}$ peak position")
    ax.legend(fontsize=7)

    # ── right panel: tau_peak(N) ─────────────────────────────────────────────
    ax = axes[1]
    for i in range(len(N)):
        ax.scatter(N[i], tp[i], color=color(int(N[i])), s=25, zorder=3,
                   label=nlabel(int(N[i])))

    if np.isfinite(g1) and np.isfinite(ta):
        xfit = np.geomspace(N.min() * 0.7, N.max() * 1.5, 300)
        yfit = ta * xfit ** g1
        safe_loglog(ax, xfit, yfit, "--", color="tab:red", lw=1.5,
                    label=rf"$\gamma_1/\nu={g1:.4f}$, $R^2={tr2:.3f}$")

    note = (r"$\gamma_1/\nu \approx 0$: expected" "\n"
            r"(lifespan peak $\approx$ const.)")
    ax.text(0.05, 0.08, note, transform=ax.transAxes,
            fontsize=10, va="bottom", color="0.4")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\langle\tau\rangle_\mathrm{peak}$")
    ax.set_title(rf"$\gamma={gamma:.1f}$ peak height")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"part2_fss_gamma_{gamma:.1f}.{ext}"),
                    dpi=300 if ext == "png" else None)
    plt.close()

# ── Figure 3: data collapse ───────────────────────────────────────────────────

def make_collapse_plot(gamma, gamma_df, summary_row):
    """
    FIX-7: for gamma<3 uses lambda_c=0 and plain tau on y-axis (gamma1/nu~0).
    FIX-1: rows with P_end > P_END_EXCL are dropped.
    Two sub-panels: left = tau collapse, right = P_end collapse (if available).
    """
    lc  = float(summary_row["lambda_c"])       if pd.notna(summary_row["lambda_c"])       else np.nan
    inv = float(summary_row["inv_nu"])         if pd.notna(summary_row["inv_nu"])         else np.nan
    g1  = float(summary_row["gamma1_over_nu"]) if pd.notna(summary_row["gamma1_over_nu"]) else 0.0
    bnu = float(summary_row["beta_over_nu"])   if pd.notna(summary_row["beta_over_nu"])   else np.nan

    if not (np.isfinite(lc) and np.isfinite(inv)):
        return

    has_pend = np.isfinite(bnu)
    ncols = 2 if has_pend else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2), squeeze=False)

    for N, sdf in sorted(gamma_df.groupby("N"), key=lambda z: int(z[0])):
        sdf = sdf.loc[sdf["P_end"] <= P_END_EXCL].copy()  # FIX-1
        if len(sdf) < 2:
            continue
        lam = sdf["lambda"].to_numpy(float)
        tau = sdf["tau"].to_numpy(float)
        pen = sdf["P_end"].to_numpy(float)
        Nf  = float(N)

        x     = (lam - lc) * Nf ** inv
        y_tau = tau / (Nf ** g1)
        mask  = np.isfinite(x) & np.isfinite(y_tau) & (y_tau > 0)
        if mask.sum() >= 2:
            axes[0, 0].plot(x[mask], y_tau[mask],
                            color=color(int(N)), lw=1.2, label=nlabel(int(N)))

        if has_pend:
            y_pend = pen * (Nf ** bnu)
            mask2  = np.isfinite(x) & np.isfinite(y_pend) & (y_pend > 0)
            if mask2.sum() >= 2:
                axes[0, 1].plot(x[mask2], y_pend[mask2],
                                color=color(int(N)), lw=1.2, label=nlabel(int(N)))

    ax = axes[0, 0]
    ax.set_xlabel(r"$(\lambda - \lambda_c)\, N^{1/\nu}$")
    ylabel_tau = (r"$\langle\tau\rangle$" if abs(g1) < 0.01
                  else r"$\langle\tau\rangle / N^{\gamma_1/\nu}$")
    ax.set_ylabel(ylabel_tau)
    ax.set_title(rf"$\gamma={gamma:.1f}$ $\tau$ collapse")
    ax.legend(fontsize=7, ncol=2)
    ax.text(0.97, 0.05,
            rf"$\lambda_c={lc:.4f}$,  $1/\nu={inv:.3f}$"
            + (rf",  $\gamma_1/\nu={g1:.4f}$" if abs(g1) > 0.001 else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.4")

    if has_pend:
        ax2 = axes[0, 1]
        ax2.set_xlabel(r"$(\lambda - \lambda_c)\, N^{1/\nu}$")
        ax2.set_ylabel(r"$P_\mathrm{end}\, N^{\beta/\nu}$")
        ax2.set_title(rf"$\gamma={gamma:.1f}$  —  $P_\mathrm{{end}}$ collapse")
        ax2.legend(fontsize=7, ncol=2)
        ax2.text(0.97, 0.05, rf"$\beta/\nu={bnu:.3f}$",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=7, color="0.4")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"part2_collapse_gamma_{gamma:.1f}.{ext}"),
                    dpi=300 if ext == "png" else None)
    plt.close()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    data    = load_raw_results()
    summary = pd.read_csv(os.path.join(PROC_DIR, "part2_critical_exponents.csv"))

    for gamma, gamma_df in data.groupby("gamma"):
        gamma_df = gamma_df.sort_values(["N","lambda"]).reset_index(drop=True)
        srow     = summary.loc[summary["gamma"] == gamma].iloc[0]
        peak_df  = pd.read_csv(
            os.path.join(PROC_DIR, f"part2_peaks_gamma_{gamma:.1f}.csv"))

        make_tau_plot(gamma, gamma_df, peak_df)
        make_fss_plot(gamma, peak_df, srow)
        make_collapse_plot(gamma, gamma_df, srow)

    print("Figures written to", FIG_DIR)

if __name__ == "__main__":
    main()