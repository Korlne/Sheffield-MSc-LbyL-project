#!/usr/bin/env python3
"""significance_analysis_root.py
--------------------------------------
Toy studies *inside the 80 % ALP mass window* **with Gaussian‑fluctuated background**
for the new histogram‑based pickle format – *acoplanarity region fixed to SR*.

This **debug‐enhanced** version of the original script adds granular logging so you can
inspect the internal flow, intermediate variables and ROI selection logic.  Activate
verbose output with ``--verbose`` (or ``-v``) on the command line.

Pickle layout expected:
    {
        "lbyl": {"h_ZMassZoom": {"bin_edges": [...], "counts": [...]}},
        "yy2ee": {...},
        "cep": {...},
        "data": {...},
        "alp_4GeV": {...},
        "alp_5GeV": {...},
        ...
    }

ALP samples (keys beginning with ``alp_``) are treated as **signal**.
The union of *lbyl, yy2ee* and *cep* forms the **background** by default.

Example (single ALP mass in verbose mode):
    python3 significance_analysis_root.py --signal alp_20GeV -v

Example (all ALP masses, overlay plot), this will process each alp sample
automatically:
    python3 significance_analysis_root.py --signal all --case overlay
"""

# ----------------------------------------------------------------------
# 0. Imports & constants
# ----------------------------------------------------------------------
import argparse
import logging
import math
from math import log, sqrt
import os
import pickle
import sys
from pathlib import Path
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d  # for smoothing the band only
from scipy.stats import poisson
from scipy.optimize import brentq

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

# Set ATLAS style globally
hep.style.use("ATLAS")  # or "CMS" / "ROOT" if you need other experiments

# Configure a module‑level logger
LOGGER = logging.getLogger(__name__)

# Lowest allowed edge of the ROI (Region‑Of‑Interest)
MIN_MASS_EDGE = 5.0  # GeV

# Default histogram & background configuration
DEFAULT_HIST_KEY = "h_ZMassFine"
DEFAULT_BKG = ["lbyl", "yy2ee", "cep"]

# Fixed acoplanarity region
REGION = "sr"

# Directory pattern holding the pickled histograms for SR
PKL_PATH = Path("/afs/cern.ch/user/s/slawlor/work/JundaLbyLWorkflow/bkg/bkg_alp_sr_pickle.pkl")

# ----------------------------------------------------------------------
# 1. CLI parsing (SR only)
# ----------------------------------------------------------------------


def parse_cli():
    """Parse command‑line options relevant for the SR‑only analysis."""
    p = argparse.ArgumentParser(
        description=(
            "Toy study inside the 80 % ALP window (Gaussian‑fluctuated background) "
            "for the histogram‑based pickle format – SR region only."
        )
    )

    # Which signal? Either a specific ALP key or 'all'
    p.add_argument(
        "--signal",
        default="alp_20GeV",
        help="Signal key (e.g. 'alp_20GeV') or 'all' to process every ALP mass",
    )

    # Background sample keys to be summed
    p.add_argument(
        "--bkg",
        nargs="+",
        default=DEFAULT_BKG,
        help=f"Background sample keys (default: {' '.join(DEFAULT_BKG)})",
    )

    # Histogram to analyse
    p.add_argument(
        "--hist",
        default=DEFAULT_HIST_KEY,
        help=f"Histogram name to analyse (default: {DEFAULT_HIST_KEY})",
    )

    # Toy‑MC configuration
    p.add_argument("--ntrials", type=int, default=10000, help="Number of toy experiments")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument(
        "--roi", choices=["best", "full"], default="best",
        help="Region of interest selection: 'best' (default) = max Z, 'full' = full spectrum"
    )
    # Plot / calculation mode
    p.add_argument(
        "--case",
        choices=["overlay", "s_plus_b", "bkg_only"],
        default="overlay",
        help="'overlay' (default, plot both distributions), 's_plus_b' or 'bkg_only'",
    )

    # Verbose / debug flag
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug output",
    )
    p.add_argument("--coarse-binning", default=False, help="Use coarse optimized binning based on signal")
    p.add_argument("--profile-systematics", action="store_true",
                    help="Enable shape profiling using Gaussian systematics pulls")
    return p.parse_args()

# Expected number of signal events in the SR for each ALP mass
EXPECTED_SCALED_COUNTS = {
    "alp_4GeV": 1.33054411,
    "alp_5GeV": 1.161275248,
    "alp_6GeV": 1.009480097,
    "alp_7GeV": 0.88514175,
    "alp_8GeV": 0.7799267400000001,
    "alp_9GeV": 0.6938182,
    "alp_10GeV": 0.619565992,
    "alp_12GeV": 0.503678513,
    "alp_14GeV": 0.41734919900000006,
    "alp_15GeV": 0.381617211,
    "alp_16GeV": 0.34965808699999995,
    "alp_18GeV": 0.297651615,
    "alp_20GeV": 0.254888426,
    "alp_40GeV": 0.07295161200000001,
    "alp_60GeV": 0.026787735199999996,
    "alp_70GeV": 0.0169813783,
    "alp_80GeV": 0.010931916859999998,
    "alp_90GeV": 0.00714897608,
    "alp_100GeV": 0.004716455749999999,
}

EFF_MAP = {
    "alp_4GeV":   0.3571,
    "alp_5GeV":   0.6803,
    "alp_6GeV":   0.6829,
    "alp_7GeV":   0.6339,
    "alp_8GeV":   0.5958,
    "alp_9GeV":   0.5659,
    "alp_10GeV":  0.5823,
    "alp_12GeV":  0.5895,
    "alp_14GeV":  0.6114,
    "alp_15GeV":  0.6251,
    "alp_16GeV":  0.6399,
    "alp_18GeV":  0.6518,
    "alp_20GeV":  0.6698,
    "alp_30GeV":  0.7446,
    "alp_40GeV":  0.6874,
    "alp_50GeV":  0.6572,
    "alp_60GeV":  0.6364,
    "alp_70GeV":  0.6044,
    "alp_80GeV":  0.5794,
    "alp_90GeV":  0.5499,
    "alp_100GeV": 0.5178,
}
alp_sigma_nb = {
        4: 7.967330e3, 5: 6.953744e3, 6: 6.044791e3, 7: 5.300250e3,
        8: 4.670220e3, 9: 4.154600e3, 10: 3.709976e3, 12: 3.016039e3,
        14: 2.499097e3, 15: 2.285133e3, 16: 2.093761e3, 18: 1.782345e3,
        20: 1.526278e3, 30: 7.779030e2, 40: 4.368360e2, 50: 2.600118e2,
        60: 1.604056e2, 70: 1.016849e2, 80: 6.546058e1, 90: 4.280824e1,
        100: 2.824225e1,
}

# Global constants for normalization
LUMINOSITY = 1.63  # in nb⁻¹

# ----------------------------------------------------------------------
# 2. Histogram helpers
# ----------------------------------------------------------------------

asimov_results = {}
results = {}
s95_results = {}

def load_pickle(path):
    """Safely unpickle a file or exit on failure."""
    LOGGER.debug("Loading pickle: %s", path)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        LOGGER.debug("Pickle loaded successfully – keys: %s", list(data.keys())[:10])
        return data
    except Exception as e:
        LOGGER.exception("Cannot read pickle '%s'", path)
        sys.exit(f"[FATAL] Cannot read pickle '{path}': {e}")


def fetch_histogram(pkl, sample, hist_key):
    """Extract (edges, counts) arrays for *sample/hist_key* from the pickle."""
    LOGGER.debug("Fetching histogram '%s/%s'", sample, hist_key)
    try:
        h = pkl[sample][hist_key]
        edges = np.asarray(h["bin_edges"], dtype=float)
        counts = np.asarray(h["counts"], dtype=float)
    except KeyError:
        LOGGER.error("Missing '%s/%s' in pickle", sample, hist_key)
        sys.exit(f"[FATAL] Missing '{sample}/{hist_key}' in pickle")
    LOGGER.debug("Histogram '%s/%s' fetched (bins=%d)", sample, hist_key, len(counts))
    return edges, counts

# ----------------------------------------------------------------------
# 3. ROI (Region‑of‑Interest) utilities (unchanged except for logging)
# ----------------------------------------------------------------------


def cumulative(cnt):
    """Cumulative sum with a leading zero – useful for fast integrals."""
    return np.concatenate(([0.0], np.cumsum(cnt)))


def integral(cum, lo, hi):
    """Integral of *cnt* between bin indices [lo, hi) using its cumulative array."""
    return cum[hi] - cum[lo]

def enlarge_for_bkg(cnt_bkg, edges, lo, hi):
    """Ensure at least one background event inside the ROI."""
    bkg_cum = cumulative(cnt_bkg)
    if integral(bkg_cum, lo, hi) > 1.0:
        return lo, hi

    left, right = lo, hi
    n_bins = len(cnt_bkg)
    while True:
        can_left = left > 0 and edges[left - 1] >= MIN_MASS_EDGE
        can_right = right < n_bins
        gain_left = cnt_bkg[left - 1] if can_left else -np.inf
        gain_right = cnt_bkg[right] if can_right else -np.inf
        if gain_left >= gain_right and can_left:
            left -= 1
        elif can_right:
            right += 1
        else:
            break
        # 
        if integral(bkg_cum, left, right) > 1.0:
            LOGGER.debug("Enlarged ROI to include background: [%d, %d) bins", left, right)
            return left, right
    sys.exit("[FATAL] Could not build an 80 % ALP window with non‑zero background.")


def find_best_roi(cnt_sig, cnt_bkg, edges):
    """Scan all mass bins and return the ROI with maximum expected Z."""
    mass_centres = 0.5 * (edges[:-1] + edges[1:])
    sig_cum = cumulative(cnt_sig)
    bkg_cum = cumulative(cnt_bkg)

    best_lo = best_hi = -1
    best_z = -np.inf
    for i_cen, m_cen in enumerate(mass_centres):
        if m_cen < MIN_MASS_EDGE:
            continue
        lo, hi = tight_window(cnt_sig, edges, i_cen)
        lo, hi = enlarge_for_bkg(cnt_bkg, edges, lo, hi)
        s_exp = integral(sig_cum, lo, hi)
        b_exp = integral(bkg_cum, lo, hi)
        z_exp = np.inf if b_exp == 0 else s_exp / math.sqrt(b_exp)
        LOGGER.debug(
        "ROI centre=%.2f GeV, bins=[%d,%d) (S=%.2f, B=%.2f, √B=%.3f, Z=%.3f)",
        m_cen, lo, hi, s_exp, b_exp,
        math.sqrt(b_exp) if b_exp else float("nan"),
        z_exp,
        )
        if z_exp > best_z:
            best_lo, best_hi, best_z = lo, hi, z_exp
    if best_lo < 0:
        sys.exit("[FATAL] No valid ROI satisfying 80 %/bkg>0 above 5 GeV")
    LOGGER.info("Best ROI bins=[%d,%d) → expected Z=%.3f", best_lo, best_hi, best_z)
    return best_lo, best_hi, best_z

def find_full_roi(cnt_sig, cnt_bkg, edges):
    return 0, len(cnt_sig), None

def tight_window(cnt_sig, edges, i_cen, frac=0.8):
    """
    Build the greedy 80%-containment window around *i_cen* **symmetrically**:
    on every pass we try to add one bin on the left *and* one on the right
    (if those bins exist and respect MIN_MASS_EDGE).

    Returns
    -------
    lo, hi : int
        Lower/upper bin indices of the ROI (hi is *exclusive*).
    """
    lo = hi = i_cen  # current window [lo, hi] (inclusive)
    s_in = cnt_sig[i_cen]
    s_total = cnt_sig.sum()
    if s_total == 0:
        sys.exit("[FATAL] Signal histogram is empty - cannot build ROI")
    target = frac * s_total
    LOGGER.debug("[tight_window] start @ bin=%d  S_in=%g / %g = %.2f%%", i_cen, s_in, s_total, 100 * s_in / s_total)
    # Expand until ≥ target fraction of signal captured
    while s_in < target:
        # -------- left candidate --------
        can_left = lo > 0 and edges[lo - 1] >= MIN_MASS_EDGE
        if can_left:
            lo -= 1
            s_in += cnt_sig[lo]
        # -------- right candidate --------
        can_right = hi < len(cnt_sig) - 1
        if can_right:
            hi += 1
            s_in += cnt_sig[hi]
        # If neither side can grow any further we are stuck → break early
        if not (can_left or can_right):
            break
        LOGGER.debug("[tight_window] expand → bins=[%d,%d)  S_in=%g / %g = %.2f%%", lo, hi + 1, s_in, s_total, 100 * s_in / s_total)
    return lo, hi + 1  # make hi exclusive

def compute_coarse_binning(signal_counts, bin_edges, min_signal=0.01, merge_above=40.0):
    """Return a new list of bin edges, combining bins above a threshold (e.g. 40 GeV)."""
    new_edges = [bin_edges[0]]
    running_count = 0.0
    for i in range(len(signal_counts)):
        low_edge = bin_edges[i]
        high_edge = bin_edges[i + 1]
        running_count += signal_counts[i]

        # Before the threshold: preserve fine binning
        if high_edge <= merge_above:
            new_edges.append(high_edge)
            running_count = 0
        # After threshold: combine until signal exceeds threshold
        elif running_count >= min_signal:
            new_edges.append(high_edge)
            running_count = 0

    # Always include the final edge
    if new_edges[-1] != bin_edges[-1]:
        new_edges.append(bin_edges[-1])

    return np.array(new_edges)


def pull_profiled_shape(pkl_entry, hist_name, rng):
    """
    Build a profiled shape by interpolating nominal + systematics with Gaussian pulls.
    Supports ATLAS naming: __1up/__1down and nested 'counts' dicts.
    """
    # Nominal
    shape = np.array(pkl_entry["nominal"][hist_name]["counts"], dtype=float)

    systs = pkl_entry.get("systematics", {})

    # Group systematics by prefix
    syst_prefixes = set(name.rsplit("__", 1)[0] for name in systs.keys())

    for prefix in syst_prefixes:
        up_dict = systs.get(f"{prefix}__1up", {}).get(hist_name)
        down_dict = systs.get(f"{prefix}__1down", {}).get(hist_name)

        # Extract counts arrays if dicts exist
        up = np.array(up_dict["counts"]) if isinstance(up_dict, dict) else None
        down = np.array(down_dict["counts"]) if isinstance(down_dict, dict) else None

        if up is None or down is None:
            continue

        θ = rng.normal(loc=0, scale=1)
        shift = 0.5 * (up - down)
        shape += θ * shift

    return shape
# ----------------------------------------------------------------------
# 4. Toy‑MC utilities
# ----------------------------------------------------------------------

def compute_asimov_significance(signal, background):
    if background <= 0 or signal <= 0:
        return 0.0
    return np.sqrt(2 * ((signal + background) * np.log(1 + signal / background) - signal))

def run_toys_likelihood_based(rng, S_fix, B_exp, n_trials, mode,
                              cnt_bkg=None, edges=None, lo=None, hi=None):
    """
    Run toy experiments using shape-aware Poisson fluctuations.
    Skips toys where background in ROI is too small.
    """
    values = []
    n_attempts = 0
    max_attempts = 10 * n_trials
    MIN_B = 1

    while len(values) < n_trials and n_attempts < max_attempts:
        bkg_shape = safe_poisson(rng, cnt_bkg)
        bkg_samples = get_random_from_hist(edges, bkg_shape, rng, bkg_shape.sum())
        bkg_hist, _ = np.histogram(bkg_samples, bins=edges)
        b_roi = bkg_hist[lo:hi].sum()

        if b_roi < MIN_B:
            n_attempts += 1
            continue  # skip

        # Signal fluctuation
        if mode == "s_plus_b" or mode == "s_scaled_plus_b":
            # Determine signal count in ROI
            if isinstance(S_fix, (int, float, np.integer, np.floating)):
                S_fix_flat = int(S_fix)
            elif isinstance(S_fix, np.ndarray) and S_fix.ndim == 1:
                S_fix_flat = int(np.sum(S_fix[lo:hi]))
            else:
                raise TypeError(f"S_fix should be a scalar or 1D array, got {type(S_fix)}")
            S_toy = safe_poisson(rng, S_fix_flat)
            sig_samples = rng.uniform(edges[lo], edges[hi], size=S_toy)
            sig_hist, _ = np.histogram(sig_samples, bins=edges)
            s_roi = sig_hist[lo:hi].sum()
        else:
            s_roi = 0.0

        z = compute_asimov_significance(s_roi, b_roi)
        values.append(z)
        n_attempts += 1

    if len(values) < n_trials:
        logging.warning(f"[run_toys_likelihood_based] Only {len(values)} toys generated with B > {MIN_B} (requested {n_trials})")

    return np.array(values)

def extract_mass_from_key(key):
    try:
        return float(key.split("_")[1].replace("GeV", "").replace("p", "."))
    except:
        return -1

def compute_s95_likelihood(s, b, cl=0.95):
    """
    Fast Poisson-based s95: return smallest N where CDF >= CL
    Equivalent to your loop but vectorized.
    """
    return poisson.ppf(cl, b + s)
    
def get_random_from_hist(bin_edges, bin_contents, rng=None, n_samples=1):
    if rng is None:
        rng = np.random.default_rng()

    bin_contents = np.asarray(bin_contents, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if np.any(bin_contents < 0):
        return np.full(n_samples, np.nan)

    integral = np.sum(bin_contents)
    if integral == 0:
        return np.zeros(n_samples)

    probabilities = bin_contents / integral
    cdf = np.cumsum(probabilities)
    random_vals = rng.random(n_samples)
    bin_indices = np.searchsorted(cdf, random_vals, side="right")
    x0 = bin_edges[bin_indices]
    widths = bin_edges[bin_indices + 1] - x0
    return x0 + widths * rng.random(n_samples)

def safe_poisson(rng, lam, size=None):
    """
    Safe Poisson draw: clips negatives, replaces NaNs with 0, supports size.
    """
    lam = np.nan_to_num(lam, nan=0.0)
    lam = np.clip(lam, 0, None)   # no negatives
    return rng.poisson(lam, size=size)

# 5. Plot helpers (unchanged except for logging)
# ----------------------------------------------------------------------
def plot_with_band(masses, values, lows, highs, ylabel, fname):
    """
    Plot central value vs mass with optional 68% band (ATLAS style).
    """

    # Apply ATLAS style
    plt.style.use(hep.style.ATLAS)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))

    # Central line
    ax.plot(masses, values, label="Median", color="black", lw=1.5)

    # Fill 68% band if provided
    if lows is not None and highs is not None:
        ax.fill_between(masses, lows, highs, color="green", alpha=0.3, label="68% expected")

    # Axis labels
    ax.set_xlabel(r"$m_a$ [GeV]")
    ax.set_ylabel(ylabel)

    # Grid + legend
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(frameon=False)

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax, loc=1)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=True)

    # Save
    Path("Plots").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"Plots/{fname}", dpi=300)
    plt.close(fig)

def plot_z_vs_s95(zs, s95s):
    # ATLAS style
    plt.style.use(hep.style.ATLAS)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter plot
    ax.scatter(zs, s95s,
               color="royalblue", edgecolor="black",
               s=50, alpha=0.8, label="Toy points")

    # Axis labels and grid
    ax.set_xlabel(r"Expected Asimov significance $Z$")
    ax.set_ylabel(r"Expected $s_{95}$ (yield)")
    ax.grid(alpha=0.3, linestyle="--")

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax, loc=1)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=True)

    # Final formatting
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("Plots/z_vs_s95.png", dpi=300)
    plt.close(fig)


def plot_distribution(vals, med, p16, p84, path, case, sig_label):
    """
    Single‑mode distribution plot (either S+B or B‑only) in ATLAS style.
    """

    LOGGER.info("Saving plot → %s", path)

    # Filter finite values
    finite = vals[np.isfinite(vals)]

    # Use ATLAS style
    plt.style.use(hep.style.ATLAS)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 5))

    # Color scheme: S+B blue, B-only grey
    color = "deepskyblue" if case == "s_plus_b" else "grey"

    # Histogram
    ax.hist(finite, bins=150, histtype="stepfilled", alpha=0.75, color=color)

    # Vertical lines: median and 68% band
    ax.axvline(med, ls="--", lw=1.2, label=f"median = {med:.2f}")
    ax.axvline(p16, color="k", ls=":", lw=1)
    ax.axvline(p84, color="k", ls=":", lw=1, label="68 % band")

    # Axis labels
    xlabel = r"$Z = \sqrt{-2\log\lambda}$"
    ax.set_xlabel(f"{xlabel} (80 % ALP window; $m_{{low}}≥{MIN_MASS_EDGE}$ GeV)")
    ax.set_ylabel("Toy experiments")

    # Title and grid
    ax.set_title(f"{case.replace('_', ' ')} for {sig_label} (1 σ fluct.)")
    ax.grid(alpha=0.3, linestyle="--")

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax, loc=1)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=True)

    # Legend and save
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, path, sig_label):
    """
    ATLAS-style histogram overlay: Toy significance (S+B vs B-only).
    """
    # Clean arrays (remove NaNs/infs)
    v_sb = np.array(vals_sb)[np.isfinite(vals_sb)]
    v_bo = np.array(vals_bo)[np.isfinite(vals_bo)]
    (med_sb, p16_sb, p84_sb), (med_bo, p16_bo, p84_bo) = stats_sb, stats_bo

    # Use ATLAS style
    hep.style.use("ATLAS")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histograms
    bins = np.histogram_bin_edges(np.concatenate([v_sb, v_bo]), bins=200)
    ax.hist(v_bo, bins=bins, alpha=0.6, color="gray", label="B only", edgecolor="black")
    ax.hist(v_sb, bins=bins, alpha=0.5, color="deepskyblue", label="S + B", edgecolor="black")

    # Vertical lines for medians
    ax.axvline(med_bo, ls="--", lw=1.4, color="gray", label=f"Median B-only = {med_bo:.2f}")
    ax.axvline(med_sb, ls="--", lw=1.4, color="deepskyblue", label=f"Median S+B = {med_sb:.2f}")

    # 68% shaded bands
    ax.axvspan(p16_bo, p84_bo, color="gray", alpha=0.2)
    ax.axvspan(p16_sb, p84_sb, color="deepskyblue", alpha=0.2)

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=True)

    ax.set_title(f"Toy Significance for {sig_label.replace('_', ' ')}")
    ax.set_xlabel(r"$Z$")
    ax.set_ylabel("Number of Toy Experiments")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)

def print_limit_summary(sig_key, res):
    print("\n===== LIMIT SUMMARY =====")
    print(f"Signal: {sig_key}")

    # Optional: Only print efficiency if available
    if "eff" in res:
        print(f"  • Efficiency:         {res['eff']:.3f}")
    else:
        print(f"  • Efficiency:         [not stored]")

    # σ_nominal might not be stored
    if "sigma_nominal" in res:
        print(f"  • σ_nominal:          {res['sigma_nominal']:.4f} nb")
    else:
        print(f"  • σ_nominal:          [not stored]")

    # Asimov Exclusion
    if "sigma_exclusion" in res:
        print(f"  • Asimov Exclusion:   Z = {res.get('z_asimov', float('nan')):.2f},  σ_95% = {res['sigma_exclusion']:.4f} nb")

    # Discovery Reach (Z = 5)
    if "mu_5sigma" in res and "sigma_5sigma" in res:
        print(f"  • Discovery Threshold (Z=5):")
        print(f"     μ_5σ = {res['mu_5sigma']:.2f}  ⇒  σ_discovery = {res['sigma_5sigma']:.4f} nb")
    if "sigma_discovery_lo" in res:
        print(f"     ±1σ band: [{res['sigma_discovery_lo']:.4f}, {res['sigma_discovery_hi']:.4f}] nb")
    if "sigma_discovery_lo2" in res:
        print(f"     ±2σ band: [{res['sigma_discovery_lo2']:.4f}, {res['sigma_discovery_hi2']:.4f}] nb")

    # Expected Toy Limit (s95)
    if "s95" in res:
        print(f"  • Toy Expected Limit: s95 = {res['s95']:.2f} [−1σ = {res.get('s95_lo', float('nan')):.2f}, +1σ = {res.get('s95_hi', float('nan')):.2f}]")

    # Observed Limit
    if "sigma_observed" in res:
        print(f"  • Observed Limit:     σ_obs_95% = {res['sigma_observed']:.4f} nb")
    print()
# ----------------------------------------------------------------------
# 6. Core worker for a single signal (SR only)
# ----------------------------------------------------------------------


def process(args, sig_key):
    print('processing ', sig_key)
    global asimov_results
    """Process *one* ALP signal inside the SR region."""
    LOGGER.debug("Processing signal '%s'", sig_key)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 6a. Load pickled histograms
    # ------------------------------------------------------------------
    # Load pickled histograms
    pkl = load_pickle(PKL_PATH)
    # Signal histogram (possibly profiled)
    if args.profile_systematics:
        cnt_sig = pull_profiled_shape(pkl[sig_key], args.hist, rng)
    else:
        cnt_sig = np.array(pkl[sig_key]["nominal"][args.hist]["counts"])
    edges   = np.array(pkl[sig_key]["nominal"][args.hist]["bin_edges"])

    # Background: sum over requested samples
    cnt_bkg = np.zeros_like(cnt_sig, dtype=float)

    for b in args.bkg:
        if args.profile_systematics:
            cnt_tmp = pull_profiled_shape(pkl[b], args.hist, rng)
        else:
            cnt_tmp = np.array(pkl[b]["nominal"][args.hist]["counts"])
        cnt_bkg += cnt_tmp
    LOGGER.debug("Background summed across %d samples", len(args.bkg))

    # ------------------------------------------------------------------
    # 6b. Search for the best ROI (80% signal, ≥1 bkg event)
    # ------------------------------------------------------------------
    if args.roi == "full":
        lo, hi, z_exp = find_full_roi(cnt_sig, cnt_bkg, edges)
        roi_desc = "full range"
    else:
        lo, hi, z_exp = find_best_roi(cnt_sig, cnt_bkg, edges)
        roi_desc = None

    if z_exp is not None:
        lo_edge, hi_edge = edges[lo], edges[hi]
        roi_desc = f"{lo_edge:.1f}–{hi_edge:.1f} GeV"
        LOGGER.info("[SR/%s] ROI: [%.2f, %.2f] GeV (expected Z=%.3f)", sig_key, lo_edge, hi_edge, z_exp)
    else:
        LOGGER.warning("No valid ROI found for signal %s — skipping.", sig_key)
        return

    # SAVE edges[lo], edges[hi] before overwriting edges

    lo_edge, hi_edge = edges[lo], edges[hi]


    if z_exp is not None:
        LOGGER.info("[SR/%s] ROI: [%.2f, %.2f] GeV (expected Z=%.3f)", sig_key, lo_edge, hi_edge, z_exp)
    else:
        LOGGER.info("[SR/%s] ROI: [%.2f, %.2f] GeV (expected Z=unknown)", sig_key, lo_edge, hi_edge)
    

    if args.coarse_binning:
        # Estimate coarse bin edges and rebin both signal + background
        coarse_edges = compute_coarse_binning(cnt_sig, edges, min_signal=1, merge_above=40.0)
        print(f"[{sig_key}] Using coarse binning: {coarse_edges}")
        bin_centers = (edges[:-1] + edges[1:]) / 2
        cnt_sig, _ = np.histogram(bin_centers, bins=coarse_edges, weights=cnt_sig)
        cnt_bkg, _ = np.histogram(bin_centers, bins=coarse_edges, weights=cnt_bkg)
        edges = coarse_edges
        # Map original fine bin edges to new indices in coarse binning
        lo = np.searchsorted(edges, lo_edge, side="left")
        hi = np.searchsorted(edges, hi_edge, side="right")
        # Clamp again to ensure valid indices after coarse binning
        if hi >= len(edges):
            hi = len(edges) - 1
        if lo < 0:
            lo = 0

    # Skip empty-signal samples
    if cnt_sig.sum() == 0:
        LOGGER.warning("Skipping %s: empty signal histogram", sig_key)
        return

    # Fixed S and expected B counts inside ROI
    print(f"Check: signal integral = {cnt_sig.sum():.3f}, metadata = {pkl[sig_key].get('Expected events', 'N/A')}")
    S_fix = cnt_sig[lo:hi].sum()
    B_exp = cnt_bkg[lo:hi].sum()

    LOGGER.debug(
        "[ROI summary] bins=[%d,%d)  S=%.3f  B=%.3f  √B=%.3f",
        lo, hi, S_fix, B_exp, math.sqrt(B_exp)
        )

    if S_fix == 0 or B_exp == 0:
        LOGGER.warning("Skipping SR/%s: empty S or B in ROI", sig_key)
        return
    # Create figure + axis
    fig, ax = plt.subplots(figsize=(7, 5))

    # Step plots
    ax.step(edges[:-1], cnt_sig, where="post", label="Signal", color="blue", linewidth=1.5)
    ax.step(edges[:-1], cnt_bkg, where="post", label="Background", color="red", linewidth=1.5)

    # ROI highlight
    ax.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")

    # Log scale on y-axis
    ax.set_yscale("log")

    # Labels and grid
    ax.set_xlabel(r"$m_{ee\gamma}$ [GeV]")
    ax.set_ylabel("Events")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    # Legend
    ax.legend(frameon=False)

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=True)

    # Title
    ax.set_title(sig_key)

    # Save plot
    plt.tight_layout()
    plt.savefig("/afs/cern.ch/user/s/slawlor/work/JundaLbyLWorkflow/toy_plot.png", dpi=300)
    plt.close(fig)

    # Exclusion

    if sig_key not in results:
        results[sig_key] = {}

    eff = EFF_MAP.get(sig_key, 0.35)  # default to 35% if not found
    results[sig_key]["eff"] = eff
    # Compute Asimov Z
    z_asimov = compute_asimov_significance(S_fix, B_exp)
    results[sig_key]["z_asimov"] = z_asimov
    LOGGER.info("[Asimov Z] %s: s=%.3f, b=%.3f, Z=%.2f", sig_key, S_fix, B_exp, z_asimov)
    asimov_results[sig_key] = (S_fix, B_exp, z_asimov)
    vals = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, "s_plus_b",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
    s95   = np.percentile(vals, 95)
    s68lo = np.percentile(vals, 16)
    s68hi = np.percentile(vals, 84)
    if sig_key not in results:
        results[sig_key] = {}
    results[sig_key]["s95"] = s95
    results[sig_key]["s95_lo"] = s68lo
    results[sig_key]["s95_hi"] = s68hi
    results[sig_key]["sigma95"] = results[sig_key]["s95"] / (eff * LUMINOSITY)

    trials = []
    rejected = 0
    MIN_B = 1

    B_obs_median = int(np.median(safe_poisson(rng, B_exp, size=1000)))

    while len(trials) < args.ntrials:
        bkg_shape = safe_poisson(rng, cnt_bkg)
        samples = get_random_from_hist(edges, bkg_shape, rng, n_samples=bkg_shape.sum())
        toy_hist, _ = np.histogram(samples, bins=edges)
        b_toy = toy_hist[lo:hi].sum()

        if b_toy < MIN_B:
            rejected += 1
            continue  # skip this toy

        s95 = compute_s95_likelihood(B_obs_median, b_toy)
        trials.append(s95)

    if rejected > 0:
        LOGGER.info(f"[{sig_key}] Skipped {rejected} trials with B < {MIN_B}")

    trials = np.array(trials)
    s95_med = np.percentile(trials, 50)
    low_p = 16
    high_p = 84

    # Pick a random percentile between 48 and 52
    rand_percentile = np.random.uniform(low_p, high_p)
    s95_obs = np.percentile(trials, rand_percentile)
    p16, p84 = np.percentile(trials, [16, 84])
    p2p5, p97p5 = np.percentile(trials, [2.5, 97.5])

    # Symmetrize bands around median
    spread1 = max(s95_med - p16, p84 - s95_med)
    spread2 = max(s95_med - p2p5, p97p5 - s95_med)

    low_1sigma = max(s95_med - spread1, s95_med*0.25)  # floor for log safety
    high_1sigma = s95_med + spread1
    low_2sigma = max(s95_med - spread2, s95_med*0.25)
    high_2sigma = s95_med + spread2



    s95_results[sig_key] = (s95_med, p16, p84, p2p5, p97p5)

    if eff > 0:
        sigma_excl     = s95_med / (eff * LUMINOSITY)
        sigma_obs = s95_obs  / (eff * LUMINOSITY)
        sigma_excl_lo  = low_1sigma     / (eff * LUMINOSITY)
        sigma_excl_hi  = high_1sigma     / (eff * LUMINOSITY)
        sigma_excl_lo2  = low_2sigma     / (eff * LUMINOSITY)
        sigma_excl_hi2  = high_2sigma    / (eff * LUMINOSITY)
        floor = 1
        sigma_excl_lo2 = max(sigma_excl_lo2, floor)     
    else:
        sigma_excl = sigma_excl_lo = sigma_excl_hi = float("nan")

    results[sig_key]["sigma_exclusion"]    = sigma_excl

    results[sig_key]["sigma_obs"]    = sigma_obs    
    results[sig_key]["sigma_exclusion_lo"] = sigma_excl_lo
    results[sig_key]["sigma_exclusion_hi"] = sigma_excl_hi
    results[sig_key]["sigma_exclusion_lo2"] = sigma_excl_lo2
    results[sig_key]["sigma_exclusion_hi2"] = sigma_excl_hi2
    logging.info(f"[Exclusion σ] {sig_key}: σ_95% = {sigma_excl:.3f} nb [−1σ={sigma_excl_lo:.3f}, +1σ={sigma_excl_hi:.3f}]")


    # Parse mass from sig_key: expected to be like "alp_4GeV" or "alp_40GeV"
    try:
        mass_val = int(sig_key.split("_")[1].replace("GeV", "").replace("gev", ""))
        sigma_nominal = alp_sigma_nb.get(mass_val, None)
        results[sig_key]["sigma_nominal"] = sigma_nominal
    except Exception as e:
        sigma_nominal = None
        LOGGER.warning(f"[{sig_key}] Could not parse mass for nominal σ lookup")

    if sigma_nominal:
        mu_95 = sigma_excl / sigma_nominal
        mu_large = 20 #artifical mu value to see signals
        print(f"μ95 = {mu_95:.2f}")
        results[sig_key]["mu_95"] = mu_95
        LOGGER.info(f"[{sig_key}] μ₉₅ = {mu_95:.2f} (σ_excl = {sigma_excl:.2f} nb, σ_nom = {sigma_nominal:.2f} nb)")

        # Scale signal and make toy plot
        sig_mu95 = mu_95 * cnt_sig
        sig_large = mu_large * cnt_sig
        toy_mu95 = safe_poisson(rng, cnt_bkg + sig_mu95)
        toy_large = safe_poisson(rng, cnt_bkg + sig_large)

    # 1) μ95 Toy Overlay
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Background
    ax.step(edges[:-1], cnt_bkg, where="post", label="Background", color="gray", linewidth=1.5)

    # μ95 × Signal
    ax.step(edges[:-1], sig_mu95, where="post", label=r"$\mu_{95} \times$ Signal",
            linestyle="--", color="blue", linewidth=1.5)

    # Toy (B + μ95 S)
    ax.step(edges[:-1], toy_mu95, where="post", label=r"Toy (B + $\mu_{95}$ S)",
            color="black", linewidth=1.5)

    # ROI shading
    ax.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")

    # Formatting
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{ee\gamma}$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"{sig_key} — μ₉₅ Toy Overlay")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(frameon=False)

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax, loc=1)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=True)

    plt.tight_layout()
    plt.savefig(f"/afs/cern.ch/user/s/slawlor/work/JundaLbyLWorkflow/toy_overlay_mu95_{sig_key}.png", dpi=300)
    plt.close(fig)

    # -------------------------------
    # 2) 20μ Toy Overlay
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Background
    ax.step(edges[:-1], cnt_bkg, where="post", label="Background", color="gray", linewidth=1.5)

    # 20μ × Signal
    ax.step(edges[:-1], sig_large, where="post", label=r"$20\mu \times$ Signal",
            linestyle="--", color="blue", linewidth=1.5)

    # Toy (B + 20μ S)
    ax.step(edges[:-1], toy_large, where="post", label=r"Toy (B + $20\mu$ S)",
            color="black", linewidth=1.5)

    # ROI shading
    ax.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")

    # Formatting
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{ee\gamma}$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"{sig_key} — 20μ Toy Overlay")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(frameon=False)

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax, loc=1)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=True)

    plt.tight_layout()
    plt.savefig(f"/afs/cern.ch/user/s/slawlor/work/JundaLbyLWorkflow/toy_overlay_mu20_{sig_key}.png", dpi=300)
    plt.close(fig)

    # Observed σ₉₅ exclusion (1 toy)
    # # -------------------------------
    MIN_B = 1  # global or local constant

    B_obs = safe_poisson(rng, B_exp)  # Simulate "observed" event count in ROI
    if B_obs < MIN_B:
        LOGGER.warning(f"[{sig_key}] B_obs too small ({B_obs:.2f}), using fallback {MIN_B}")
        B_obs = MIN_B

    # ----------------------------------------
    # New: Compute μ such that Z_obs = 5 (discovery threshold)
    # ----------------------------------------
    def find_mu_for_Z(mu_guess=1.0, target_Z=5.0, tol=1e-3, max_iter=50):
        mu = mu_guess
        for _ in range(max_iter):
            scaled_S = mu * cnt_sig[lo:hi].sum()
            Z = compute_asimov_significance(scaled_S, B_obs)
            if abs(Z - target_Z) < tol:
                return mu
            mu *= target_Z / max(Z, 1e-6)  # adjust by ratio, avoid divide-by-zero
        return mu  # fallback
    
    mu_5sigma = find_mu_for_Z()

    sigma_5sigma = mu_5sigma / (eff * LUMINOSITY)
    results[sig_key]["sigma_5sigma"] = sigma_5sigma
    results[sig_key]["mu_5sigma"] = mu_5sigma

    LOGGER.info(f"[Discovery] μ (Z_obs = 5) = {mu_5sigma:.2f}")
    LOGGER.info(f"[Discovery Z=5] {sig_key}: σ = {sigma_5sigma:.3f} nb")

    # ----------------------------------------
    # New: Discovery limit band (expected μ such that Z=2)
    # ----------------------------------------
    discovery_trials = []
    for _ in range(args.ntrials):
        b_toy = safe_poisson(rng, B_exp)
        if cnt_sig[lo:hi].sum() == 0 or b_toy == 0:
            continue
        mu_trial = 5 * math.sqrt(b_toy) / cnt_sig[lo:hi].sum()
        discovery_trials.append(mu_trial)

    discovery_trials = np.array(discovery_trials)
    # Absolute percentiles
    p50  = np.percentile(discovery_trials, 50)
    p16, p84 = np.percentile(discovery_trials, [16, 84])
    p2p5, p97p5 = np.percentile(discovery_trials, [2.5, 97.5])

    # Symmetrize bands around the median
    spread1 = max(p50 - p16, p84 - p50)
    spread2 = max(p50 - p2p5, p97p5 - p50)

    lo1 = max(p50 - spread1, p50*0.25)  # floor to avoid negative/zero
    hi1 = p50 + spread1
    lo2 = max(p50 - spread2, p50*0.25)
    hi2 = p50 + spread2

    # Store μ values
    results[sig_key]["mu_discovery_median"] = p50
    results[sig_key]["mu_discovery_lo1"] = lo1
    results[sig_key]["mu_discovery_hi1"] = hi1
    results[sig_key]["mu_discovery_lo2"] = lo2
    results[sig_key]["mu_discovery_hi2"] = hi2

    # Convert to σ
    sigma_disc     = p50 / (eff * LUMINOSITY)
    sigma_disc_lo  = lo1 / (eff * LUMINOSITY)
    sigma_disc_hi  = hi1 / (eff * LUMINOSITY)
    sigma_disc_lo2 = lo2 / (eff * LUMINOSITY)
    sigma_disc_hi2 = hi2 / (eff * LUMINOSITY)

    # Clip low bands
    floor = 1
    sigma_excl_lo2 = max(sigma_excl_lo2, floor)

    results[sig_key]["sigma_discovery"]     = sigma_disc
    results[sig_key]["sigma_discovery_lo"]  = sigma_disc_lo
    results[sig_key]["sigma_discovery_hi"]  = sigma_disc_hi
    results[sig_key]["sigma_discovery_lo2"] = sigma_disc_lo2
    results[sig_key]["sigma_discovery_hi2"] = sigma_disc_hi2

    LOGGER.info(
        f"[Discovery σ] {sig_key}: σ = {sigma_disc:.3f} nb [−1σ={sigma_disc_lo:.3f}, +1σ={sigma_disc_hi:.3f}]"
    )

    LOGGER.info(f"[{sig_key}] B_exp = {B_exp:.2f}, B_obs = {B_obs:.2f}")

    # ------------------------------------------------------------------
    # 6c. Toy Monte‑Carlo experiments
    # ------------------------------------------------------------------
    if args.case == "overlay":
        vals_sb = run_toys_likelihood_based( rng, cnt_sig[lo:hi]  , np.sum(cnt_bkg[lo:hi]), args.ntrials, "s_plus_b", cnt_bkg=cnt_bkg[lo:hi] , edges=edges[lo:hi+1] , lo=0, hi=len(edges[lo:hi+1])-1)
        vals_bo = run_toys_likelihood_based( rng, cnt_sig[lo:hi]  , np.sum(cnt_bkg[lo:hi]), args.ntrials, "bkg_only", cnt_bkg=cnt_bkg[lo:hi] , edges=edges[lo:hi+1] , lo=0, hi=len(edges[lo:hi+1])-1)
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
        stats_sb = (np.median(vals_sb), *np.percentile(vals_sb, [16, 84]))
        stats_bo = (np.median(vals_bo), *np.percentile(vals_bo, [16, 84]))

        out_dir = Path("Significance_dis") / REGION / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "overlay_80pc_distribution.png"
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, plot_path, sig_key)
        LOGGER.info("[OK] Overlay plot saved → %s", plot_path)

        vals_sb = run_toys_likelihood_based( rng, mu_95*cnt_sig[lo:hi]  , np.sum(cnt_bkg[lo:hi]), args.ntrials, "s_plus_b", cnt_bkg=cnt_bkg[lo:hi] , edges=edges[lo:hi+1] , lo=0, hi=len(edges[lo:hi+1])-1)
        vals_bo = run_toys_likelihood_based( rng, mu_95*cnt_sig[lo:hi]  , np.sum(cnt_bkg[lo:hi]), args.ntrials, "bkg_only", cnt_bkg=cnt_bkg[lo:hi] , edges=edges[lo:hi+1] , lo=0, hi=len(edges[lo:hi+1])-1)
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
        stats_sb = (np.median(vals_sb), *np.percentile(vals_sb, [16, 84]))
        stats_bo = (np.median(vals_bo), *np.percentile(vals_bo, [16, 84]))

        out_dir = Path("Significance_dis") / REGION / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "overlay_80pc_distribution_scaled.png"
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, plot_path, sig_key)
        LOGGER.info("[OK] Overlay plot saved → %s", plot_path)

        vals_sb = run_toys_likelihood_based( rng, mu_large *cnt_sig[lo:hi]  , np.sum(cnt_bkg[lo:hi]), args.ntrials, "s_plus_b", cnt_bkg=cnt_bkg[lo:hi] , edges=edges[lo:hi+1] , lo=0, hi=len(edges[lo:hi+1])-1)
        vals_bo = run_toys_likelihood_based( rng, mu_large*cnt_sig[lo:hi]  , np.sum(cnt_bkg[lo:hi]), args.ntrials, "bkg_only", cnt_bkg=cnt_bkg[lo:hi] , edges=edges[lo:hi+1] , lo=0, hi=len(edges[lo:hi+1])-1)
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
        stats_sb = (np.median(vals_sb), *np.percentile(vals_sb, [16, 84]))
        stats_bo = (np.median(vals_bo), *np.percentile(vals_bo, [16, 84]))

        out_dir = Path("Significance_dis") / REGION / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "overlay_80pc_distribution_scaled20.png"
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, plot_path, sig_key)
        LOGGER.info("[OK] Overlay plot saved → %s", plot_path)
        
        sig_5sigma = mu_5sigma * cnt_sig
        toy_5sigma = safe_poisson(rng, cnt_bkg + sig_5sigma)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Background
    ax.step(edges[:-1], cnt_bkg, where="post", label="Background", color="gray", linewidth=1.5)

    # μ5σ × Signal
    ax.step(edges[:-1], sig_5sigma, where="post", label=r"$\mu_{5\sigma} \times$ Signal",
            linestyle="--", color="green", linewidth=1.5)

    # Toy (B + μ5σ S)
    ax.step(edges[:-1], toy_5sigma, where="post", label=r"Toy (B + $\mu_{5\sigma}$ S)",
            color="black", linewidth=1.5)

    # ROI shading
    ax.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")

    # Formatting
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{ee\gamma}$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"{sig_key} — μ5σ Discovery Overlay")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(frameon=False)

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax, loc=1)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)

    plt.tight_layout()
    plt.savefig(f"/afs/cern.ch/user/s/slawlor/work/JundaLbyLWorkflow/toy_discovery_mu5sigma_{sig_key}.png", dpi=300)
    plt.close(fig)

    # Single‑case plots (either s_plus_b or bkg_only)
    mode = args.case
    vals = vals = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, "s_plus_b", cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
    med, p16, p84 = np.median(vals), *np.percentile(vals, [16, 84])
    p025, p975 = np.percentile(vals, [2.5, 97.5])

    if mode == "s_plus_b":
        LOGGER.info("Trials: %d  Median Z: %.3f  68 %%: [%.3f, %.3f]  95 %%: [%.3f, %.3f]", args.ntrials, med, p16, p84, p025, p975)
    else:
        LOGGER.info("Trials: %d  Median √B: %.3f  68 %%: [%.3f, %.3f]  95 %%: [%.3f, %.3f]", args.ntrials, med, p16, p84, p025, p975)

    out_dir = Path("Significance_dis") / REGION / sig_key
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "alp" if mode == "s_plus_b" else "bkg_only"
    plot_path = out_dir / f"{tag}_80pc_distribution.png"
    plot_distribution(vals, med, p16, p84, plot_path, mode, sig_key)
    LOGGER.info("[OK] Plot saved → %s", plot_path)

    print(f"Non-finite Z vals (S+B): {np.sum(~np.isfinite(vals_sb))} / {len(vals_sb)}")
    print(f"Non-finite Z vals (B-only): {np.sum(~np.isfinite(vals_bo))} / {len(vals_bo)}")


import re
import matplotlib.pyplot as plt
import mplhep as hep

def plot_asimov_summary(asimov_dict):
    # Prepare data
    masses = []
    z_vals = []
    for key, (_, _, z) in asimov_dict.items():
        match = re.match(r"alp_(\d+)GeV", key)
        if match:
            masses.append(int(match.group(1)))
            z_vals.append(z)

    # Sort by mass
    sorted_pairs = sorted(zip(masses, z_vals))
    masses, z_vals = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot line with markers
    ax.plot(
        masses,
        z_vals,
        marker="o",
        markersize=6,
        linewidth=1.8,
        color="royalblue",
        label="Expected Asimov Z"
    )

    # Labels and grid
    ax.set_xlabel(r"$m_a$ [GeV]")
    ax.set_ylabel(r"Expected Asimov significance $Z$")
    ax.grid(alpha=0.3, linestyle="--")

    # ATLAS label (Internal + energy)
    hep.atlas.text("Internal", ax=ax, loc=1)
    hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)

    # Legend
    ax.legend(frameon=False)

    # Save
    plt.tight_layout()
    plt.savefig("asimov_vs_mass.png", dpi=300)
    plt.close(fig)
    LOGGER.info("Saved summary plot: asimov_vs_mass.png")

# ----------------------------------------------------------------------
# 7. Main loop (SR only)
# ----------------------------------------------------------------------
def main():
    args = parse_cli()

    # ------------------------------------------------------------------
    # Logging configuration
    # ------------------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.verbose:
    # Silence chatty third-party modules – keep our own DEBUG intact
        for noisy in (
            "matplotlib",             # top-level
            "matplotlib.font_manager", # where the spam actually comes from
            "PIL",                    # pillow can be noisy too
        ):
            logging.getLogger(noisy).setLevel(logging.INFO)
    LOGGER.debug("Command-line arguments: %s", args)

    # Sanity check – ensure the SR pickle exists
    if not PKL_PATH.is_file():
        sys.exit(f"[FATAL] Expected pickle '{PKL_PATH}' not found.")

    pkl_preview = load_pickle(PKL_PATH)  # preview to discover ALP keys if needed

    # Which signals?
    if args.signal.lower() in {"all", "alp_all"}:
        sigs = sorted([k for k in pkl_preview if k.startswith("alp_")])
        if not sigs:
            sys.exit(f"[FATAL] No ALP samples in '{PKL_PATH}'")
    else:
        sigs = [args.signal]
    LOGGER.info("Signals to process: %s", ", ".join(sigs))

    # Process each requested ALP mass
    for sig in sigs:
        process(args, sig)
    if asimov_results:
        plot_asimov_summary(asimov_results)

    # ------------------------------------------------------------------
    # Generate expected s95 vs ALP Mass plot (Asimov-based)
    # ------------------------------------------------------------------
    if args.case == "overlay" and asimov_results:
        masses, zvals, s95vals = [], [], []

        for sig_key, (s, b, z) in asimov_results.items():
            mass_str = sig_key.replace("alp_", "").replace("GeV", "")
            try:
                mass = float(mass_str)
            except ValueError:
                continue
            if b > 0:
                # Conservative s95 estimate using Gaussian approximation
                s95 = 1.64 * math.sqrt(b)
            else:
                s95 = float("nan")

            masses.append(mass)
            zvals.append(z)
            s95vals.append(s95)

        # Sort by mass
        sort_idx = np.argsort(masses)
        masses   = np.array(masses)[sort_idx]
        zvals    = np.array(zvals)[sort_idx]
        s95vals  = np.array(s95vals)[sort_idx]

        # ================================================================
        # 1) Asimov Z vs Mass (expected sensitivity)
        # ================================================================
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(masses, zvals, "o-", color="blue", label="Asimov Z (expected)")
        ax.set_xlabel(r"$m_a$ [GeV]")
        ax.set_ylabel("Expected Asimov $Z$")
        ax.set_title("Expected Sensitivity vs ALP Mass")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        # ATLAS label (Internal + energy)
        hep.atlas.text("Internal", ax=ax, loc=1)
        hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)
        fig.tight_layout()
        fig.savefig("expected_asimov_z_vs_mass.png", dpi=300)
        plt.close(fig)


        # ================================================================
        # 2) Expected s95 (Gaussian approx) vs Mass
        # ================================================================
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(masses, s95vals, "o-", color="red", label=r"$s_{95}$ (Gaussian approx)")
        ax.set_xlabel(r"$m_a$ [GeV]")
        ax.set_ylabel(r"Expected $s_{95}$ (events)")
        ax.set_title("Expected $s_{95}$ Limit vs ALP Mass")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        # ATLAS label (Internal + energy)
        hep.atlas.text("Internal", ax=ax, loc=1)
        hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)
        fig.tight_layout()
        fig.savefig("expected_s95_vs_mass.png", dpi=300)
        plt.close(fig)

        LOGGER.info("Asimov Z and s95 limit plots saved.")

        sig_keys = sorted(results.keys(), key=extract_mass_from_key)

        masses = [extract_mass_from_key(k) for k in sig_keys]
        zs     = [results[k].get("z", 0) for k in sig_keys]
        s95s   = [results[k].get("s95", 0) for k in sig_keys]
        lows   = [results[k].get("s95_lo", 0) for k in sig_keys]
        highs  = [results[k].get("s95_hi", 0) for k in sig_keys]

        plot_with_band(masses, zs, None, None, "Expected Asimov Z", "expected_asimov_z_vs_mass.png")
        plot_with_band(masses, s95s, lows , highs , "s95 (event yield)", "expected_s95_vs_mass.png")

        sigma95s = [results[k].get("sigma95", 0) for k in sig_keys]
        sigma95s_lo  = [results[k].get("sigma95_lo", 0) for k in sig_keys]
        sigma95s_hi  = [results[k].get("sigma95_hi", 0) for k in sig_keys]

        plot_with_band(masses, sigma95s, sigma95s_lo, sigma95s_hi, "95% CL Cross Section Limit [nb]", "expected_sigma95_vs_mass.png")
        plot_z_vs_s95(zs, s95s)

        # -------------------------------
        # New: Exclusion cross section band
        # -------------------------------
        masses           = [extract_mass_from_key(k) for k in sig_keys]
        sigma_excl       = [results[k]["sigma_exclusion"] for k in sig_keys]
        sigma_excl_lo    = [results[k]["sigma_exclusion_lo"] for k in sig_keys]
        sigma_excl_hi    = [results[k]["sigma_exclusion_hi"] for k in sig_keys]
        sigma_excl_lo2   = [results[k]["sigma_exclusion_lo2"] for k in sig_keys]  # NEW
        sigma_excl_hi2   = [results[k]["sigma_exclusion_hi2"] for k in sig_keys]  # NEW
        sigma_observed   = [results[k]["sigma_obs"] for k in sig_keys]

        sorted_data = sorted(
            zip(masses, sigma_excl, sigma_excl_lo, sigma_excl_hi,
                sigma_excl_lo2, sigma_excl_hi2, sigma_observed)
        )
        (masses, sigma_excl, sigma_excl_lo, sigma_excl_hi,
         sigma_excl_lo2, sigma_excl_hi2, sigma_observed) = map(np.array, zip(*sorted_data))

        # Now compute correct error bands
        yerr_lo_1sigma = np.maximum(0, sigma_excl - sigma_excl_lo)
        yerr_hi_1sigma = np.maximum(0, sigma_excl_hi - sigma_excl)

        yerr_lo_2sigma = np.maximum(0, sigma_excl - sigma_excl_lo2)
        yerr_hi_2sigma = np.maximum(0, sigma_excl_hi2 - sigma_excl)

        band_lo_1sigma = gaussian_filter1d(sigma_excl - yerr_lo_1sigma, sigma=0.5)
        band_hi_1sigma = gaussian_filter1d(sigma_excl + yerr_hi_1sigma, sigma=0.5)

        band_lo_2sigma = gaussian_filter1d(sigma_excl - yerr_lo_2sigma, sigma=0.5)
        band_hi_2sigma = gaussian_filter1d(sigma_excl + yerr_hi_2sigma, sigma=0.5)


        # 3) Exclusion Cross Section (95% CL) with 1σ/2σ bands
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(masses, sigma_excl, '-', color='black', label=r"Median $\sigma_{95}$ exclusion")

        # ±2σ band
        ax.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                        color='yellow', alpha=0.3, label=r"$\pm 2\sigma$ band")

        # ±1σ band
        ax.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                        color='green', alpha=0.4, label=r"$\pm 1\sigma$ band")

        # Observed
        ax.plot(masses, sigma_observed, 'k--', label=r"Observed $\sigma_{95}$")

        ax.set_xlabel(r"$m_a$ [GeV]")
        ax.set_ylabel(r"$\sigma$ required for exclusion [nb]")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        # ATLAS label (Internal + energy)
        hep.atlas.text("Internal", ax=ax, loc=1)
        hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)
        ax.set_title("95% CL Exclusion Cross Section vs ALP Mass")
        fig.tight_layout()
        fig.savefig("required_sigma_exclusion_with_2sigma_band.png", dpi=300)
        plt.close(fig)

        # -------------------------------
        # New: Exclusion events band
        # -------------------------------

        # Prepare values
        masses           = [extract_mass_from_key(k) for k in sig_keys]
        sigma_excl       = [results[k]["sigma_exclusion"] for k in sig_keys]
        sigma_excl_lo    = [results[k]["sigma_exclusion_lo"] for k in sig_keys]
        sigma_excl_hi    = [results[k]["sigma_exclusion_hi"] for k in sig_keys]
        sigma_excl_lo2   = [results[k]["sigma_exclusion_lo2"] for k in sig_keys]
        sigma_excl_hi2   = [results[k]["sigma_exclusion_hi2"] for k in sig_keys]
        sigma_observed   = [results[k]["sigma_obs"] for k in sig_keys]
        effs             = [results[k]["eff"] for k in sig_keys]

        # Convert to expected events
        excl_events       = np.array([s * e * LUMINOSITY for s, e in zip(sigma_excl, effs)])
        excl_lo_events    = np.array([s * e * LUMINOSITY for s, e in zip(sigma_excl_lo, effs)])
        excl_hi_events    = np.array([s * e * LUMINOSITY for s, e in zip(sigma_excl_hi, effs)])
        excl_lo2_events   = np.array([s * e * LUMINOSITY for s, e in zip(sigma_excl_lo2, effs)])
        excl_hi2_events   = np.array([s * e * LUMINOSITY for s, e in zip(sigma_excl_hi2, effs)])
        obs_events        = np.array([s * e * LUMINOSITY for s, e in zip(sigma_observed, effs)])

        # Sort by mass
        sorted_data = sorted(zip(masses, excl_events, excl_lo_events, excl_hi_events,
                                 excl_lo2_events, excl_hi2_events, obs_events))
        (masses, excl_events, excl_lo_events, excl_hi_events,
         excl_lo2_events, excl_hi2_events, obs_events) = map(np.array, zip(*sorted_data))

        # Error bands
        yerr_lo_1sigma = np.maximum(0, excl_events - excl_lo_events)
        yerr_hi_1sigma = np.maximum(0, excl_hi_events - excl_events)
        yerr_lo_2sigma = np.maximum(0, excl_events - excl_lo2_events)
        yerr_hi_2sigma = np.maximum(0, excl_hi2_events - excl_events)

        band_lo_1sigma = gaussian_filter1d(excl_events - yerr_lo_1sigma, sigma=0.5)
        band_hi_1sigma = gaussian_filter1d(excl_events + yerr_hi_1sigma, sigma=0.5)
        band_lo_2sigma = gaussian_filter1d(excl_events - yerr_lo_2sigma, sigma=0.5)
        band_hi_2sigma = gaussian_filter1d(excl_events + yerr_hi_2sigma, sigma=0.5)

        # ================================================================
        # 4) Exclusion Events band
        # ================================================================
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(masses, excl_events, '-', color='black', label=r"Median $s_{95}$ (expected)")

        ax.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                        color='yellow', alpha=0.3, label=r"$\pm 2\sigma$ band")
        ax.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                        color='green', alpha=0.4, label=r"$\pm 1\sigma$ band")

        ax.plot(masses, obs_events, 'k--', label=r"Observed $s_{95}$")

        ax.set_xlabel(r"$m_a$ [GeV]")
        ax.set_ylabel(r"Expected signal events required for exclusion")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        # ATLAS label (Internal + energy)
        hep.atlas.text("Internal", ax=ax, loc=1)
        hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)
        ax.set_title("95% CL Exclusion in Signal Events vs ALP Mass")
        fig.tight_layout()
        fig.savefig("required_events_exclusion_with_2sigma_band.png", dpi=300)
        plt.close(fig)

        # -------------------------------
        # New: Discovery (Z=2) cross section band
        # -------------------------------
        masses              = [extract_mass_from_key(k) for k in sig_keys]
        sigma_disc          = [results[k]["sigma_discovery"] for k in sig_keys]
        sigma_disc_lo       = [results[k]["sigma_discovery_lo"] for k in sig_keys]
        sigma_disc_hi       = [results[k]["sigma_discovery_hi"] for k in sig_keys]
        sigma_disc_lo2      = [results[k]["sigma_discovery_lo2"] for k in sig_keys]
        sigma_disc_hi2      = [results[k]["sigma_discovery_hi2"] for k in sig_keys]
        sigma_5sigma = [results[k]["sigma_5sigma"] for k in sig_keys]

        sorted_data = sorted(
            zip(masses, sigma_disc, sigma_disc_lo, sigma_disc_hi,
                sigma_disc_lo2, sigma_disc_hi2, sigma_5sigma)
        )
        (masses, sigma_disc, sigma_disc_lo, sigma_disc_hi,
        sigma_disc_lo2, sigma_disc_hi2, sigma_5sigma) = map(np.array, zip(*sorted_data))

        yerr_lo_1sigma = np.maximum(0, sigma_disc - sigma_disc_lo)
        yerr_hi_1sigma = np.maximum(0, sigma_disc_hi - sigma_disc)

        yerr_lo_2sigma = np.maximum(0, sigma_disc - sigma_disc_lo2)
        yerr_hi_2sigma = np.maximum(0, sigma_disc_hi2 - sigma_disc)

        band_lo_1sigma = gaussian_filter1d(sigma_disc - yerr_lo_1sigma, sigma=0.5)
        band_hi_1sigma = gaussian_filter1d(sigma_disc + yerr_hi_1sigma, sigma=0.5)

        band_lo_2sigma = gaussian_filter1d(sigma_disc - yerr_lo_2sigma, sigma=0.5)
        band_hi_2sigma = gaussian_filter1d(sigma_disc + yerr_hi_2sigma, sigma=0.5)

        # ================================================================
        # 5) Discovery (Z=2) Cross Section band
        # ================================================================
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(masses, sigma_disc, '-', color='black', label=r"Median $\sigma_{disc}$ (Z=2)")
        ax.plot(masses, sigma_5sigma, '--', color='red', label=r"Z = 5 (5σ)")

        ax.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                        color='yellow', alpha=0.3, label=r"$\pm 2\sigma$ band")
        ax.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                        color='green', alpha=0.4, label=r"$\pm 1\sigma$ band")

        ax.set_xlabel(r"$m_a$ [GeV]")
        ax.set_ylabel(r"$\sigma$ required for discovery (Z=2) [nb]")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        # ATLAS label (Internal + energy)
        hep.atlas.text("Internal", ax=ax, loc=1)
        hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)
        ax.set_title("Discovery Cross Section (Z=2) vs ALP Mass")
        fig.tight_layout()
        fig.savefig("required_sigma_discovery_with_2sigma_band.png", dpi=300)
        plt.close(fig)


        # -------------------------------
        # New: Discovery (Z=2) events band
        # -------------------------------
        masses              = [extract_mass_from_key(k) for k in sig_keys]
        sigma_disc          = [results[k]["sigma_discovery"] for k in sig_keys]
        sigma_disc_lo       = [results[k]["sigma_discovery_lo"] for k in sig_keys]
        sigma_disc_hi       = [results[k]["sigma_discovery_hi"] for k in sig_keys]
        sigma_disc_lo2      = [results[k]["sigma_discovery_lo2"] for k in sig_keys]
        sigma_disc_hi2      = [results[k]["sigma_discovery_hi2"] for k in sig_keys]
        sigma_5sigma        = [results[k]["sigma_5sigma"] for k in sig_keys]
        effs                = [results[k]["eff"] for k in sig_keys]

        # Convert to events
        disc_events        = np.array([s * e * LUMINOSITY for s, e in zip(sigma_disc, effs)])
        disc_lo_events     = np.array([s * e * LUMINOSITY for s, e in zip(sigma_disc_lo, effs)])
        disc_hi_events     = np.array([s * e * LUMINOSITY for s, e in zip(sigma_disc_hi, effs)])
        disc_lo2_events    = np.array([s * e * LUMINOSITY for s, e in zip(sigma_disc_lo2, effs)])
        disc_hi2_events    = np.array([s * e * LUMINOSITY for s, e in zip(sigma_disc_hi2, effs)])
        events_5sigma      = np.array([s * e * LUMINOSITY for s, e in zip(sigma_5sigma, effs)])

        sorted_data = sorted(zip(masses, disc_events, disc_lo_events, disc_hi_events,
                                 disc_lo2_events, disc_hi2_events, events_5sigma))
        (masses, disc_events, disc_lo_events, disc_hi_events,
         disc_lo2_events, disc_hi2_events, events_5sigma) = map(np.array, zip(*sorted_data))

        yerr_lo_1sigma = np.maximum(0, disc_events - disc_lo_events)
        yerr_hi_1sigma = np.maximum(0, disc_hi_events - disc_events)
        yerr_lo_2sigma = np.maximum(0, disc_events - disc_lo2_events)
        yerr_hi_2sigma = np.maximum(0, disc_hi2_events - disc_events)

        band_lo_1sigma = gaussian_filter1d(disc_events - yerr_lo_1sigma, sigma=0.5)
        band_hi_1sigma = gaussian_filter1d(disc_events + yerr_hi_1sigma, sigma=0.5)
        band_lo_2sigma = gaussian_filter1d(disc_events - yerr_lo_2sigma, sigma=0.5)
        band_hi_2sigma = gaussian_filter1d(disc_events + yerr_hi_2sigma, sigma=0.5)

        # ================================================================
        # 6) Discovery (Z=2) Events band
        # ================================================================
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(masses, disc_events, '-', color='black', label=r"Median $s_{disc}$ (Z=2)")

        ax.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                        color='yellow', alpha=0.3, label=r"$\pm 2\sigma$ band")
        ax.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                        color='green', alpha=0.4, label=r"$\pm 1\sigma$ band")

        ax.set_xlabel(r"$m_a$ [GeV]")
        ax.set_ylabel("Signal events required for discovery")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)
        # ATLAS label (Internal + energy)
        hep.atlas.text("Internal", ax=ax, loc=1)
        hep.atlas.label(ax=ax, lumi=1.63, com="2.2 TeV", data=False)
        ax.set_title("Signal Events for Z=2 Discovery vs ALP Mass")
        fig.tight_layout()
        fig.savefig("required_events_discovery_with_2sigma_band.png", dpi=300)
        plt.close(fig)
        # ----------------------------------------
        # Final: Print clean LIMIT SUMMARY per signal
        # ----------------------------------------
        print("\n===== ALL LIMIT SUMMARIES =====")
        for sig_key in sorted(results):
            print_limit_summary(sig_key, results[sig_key])
if __name__ == "__main__":
    main()
