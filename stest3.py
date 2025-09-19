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
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d  # for smoothing the band only


import matplotlib.pyplot as plt
import numpy as np

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
PKL_PATH = Path("/home/jtong/lbyl/bkg/bkg_alp_sr_pickle.pkl")

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
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian width scale (σ factor)")
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
    p.add_argument("--coarse-binning", default=True, help="Use coarse optimized binning based on signal")
    p.add_argument("--profile-systematics", action="store_true",
                    help="Enable shape profiling using Gaussian systematics pulls")
    p.add_argument("--use-cls", action="store_true",
                    help="Use CLs-based exclusion instead of Z-based")
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
    "alp_4GeV": 0.28,
    "alp_5GeV": 0.30,
    "alp_6GeV": 0.32,
    "alp_7GeV": 0.33,
    "alp_8GeV": 0.34,
    "alp_9GeV": 0.35,
    "alp_10GeV": 0.36,
    "alp_12GeV": 0.37,
    "alp_14GeV": 0.38,
    "alp_15GeV": 0.38,
    "alp_16GeV": 0.38,
    "alp_18GeV": 0.39,
    "alp_20GeV": 0.39,
    "alp_40GeV": 0.40,
    "alp_60GeV": 0.40,
    "alp_70GeV": 0.40,
    "alp_80GeV": 0.40,
    "alp_90GeV": 0.40,
    "alp_100GeV": 0.40,
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

    Args:
        pkl_entry (dict): e.g. pkl["alp_80GeV"]
        hist_name (str): histogram key, e.g. "h_ZMassFine"
        rng: numpy random generator

    Returns:
        np.ndarray: profiled histogram (signal or background)
    """
    # Start with nominal
    shape = np.array(pkl_entry["nominal"][hist_name]["counts"], dtype=float)

    # Profile systematics
    systs = pkl_entry.get("systematics", {})
    for syst_name in set(k.rsplit("_", 1)[0] for k in systs):
        up = systs.get(f"{syst_name}_up", {}).get(hist_name)
        down = systs.get(f"{syst_name}_down", {}).get(hist_name)

        if up is None or down is None:
            continue

        θ = rng.normal(loc=0, scale=1)
        shift = 0.5 * (np.array(up) - np.array(down))
        shape += θ * shift

    return shape

# ----------------------------------------------------------------------
# 4. Toy‑MC utilities
# ----------------------------------------------------------------------


def compute_likelihood_z(s, b, epsilon=1e-9):
    """
    Compute Z from likelihood ratio assuming s+b hypothesis.
    """
    if b <= 0:
        return 0.0
    term = (s + b) * log(1 + s / (b + epsilon)) - s
    return sqrt(2 * term) if term > 0 else 0.0

def compute_asimov_significance(signal, background):
    if background <= 0 or signal <= 0:
        return 0.0
    return np.sqrt(2 * ((signal + background) * np.log(1 + signal / background) - signal))

def draw_bg(rng, b_exp, sigma_scale):
    """Gaussian‑fluctuated background (≥0)."""
    std = math.sqrt(b_exp) * sigma_scale
    return max(rng.normal(b_exp, std), 0.0)


def run_toys(rng, S_fix, B_exp, n_trials, sigma, mode):
    """
    Generate toy values of Z = S / sqrt(B) under two hypotheses.

    - "bkg_only": fluctuate B only, fixed S used in numerator
    - "s_plus_b": fluctuate both S and B
    Skips toys with B = 0
    """
    results = []
    n_attempts = 0
    max_attempts = 20 * n_trials
    MIN_B = 1

    while len(results) < n_trials and n_attempts < max_attempts:
        B_toy = rng.poisson(B_exp)
        if B_toy < MIN_B:
            n_attempts += 1
            continue  # skip zero-background toys

        if mode == "bkg_only":
            Z = S_fix / math.sqrt(B_toy)
        elif mode == "s_plus_b":
            S_toy = rng.poisson(S_fix)
            Z = S_toy / math.sqrt(B_toy)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        results.append(Z)
        n_attempts += 1

    if len(results) < n_trials:
        logging.warning(f"[run_toys] Only {len(results)} toys generated with B > {MIN_B} (requested {n_trials})")

    return np.array(results)


def run_toys_likelihood_based(rng, S_fix, B_exp, n_trials, sigma, mode,
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
        bkg_shape = rng.poisson(cnt_bkg)
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
            S_toy = rng.poisson(S_fix_flat)
            sig_samples = rng.uniform(edges[lo], edges[hi], size=S_toy)
            sig_hist, _ = np.histogram(sig_samples, bins=edges)
            s_roi = sig_hist[lo:hi].sum()
        else:
            s_roi = 0.0

        z = compute_likelihood_z(s_roi, b_roi, sigma)
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
    # Simplified CLs-style s95 calculator using Poisson stats
    from scipy.stats import poisson
    limit = 0
    while poisson.cdf(limit, b + s) < cl:
        limit += 1
    return limit

def get_required_sigma(target_Z, eff, lumi, bkg, tol=1e-3, max_sigma=1e9):
    """
    Scale cross section σ until s = σ * eff * lumi yields Z ≥ target_Z.
    Uses binary search.
    """
    low, high = 0.0, max_sigma
    for _ in range(100):  # binary search loop
        mid = 0.5 * (low + high)
        s = mid * eff * lumi
        Z = s / np.sqrt(bkg) if bkg > 0 else 0
        if abs(Z - target_Z) < tol:
            return mid
        if Z < target_Z:
            low = mid
        else:
            high = mid
    return mid
    
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

def convert_sigma_to_events(sigma, eff, lumi):
    return sigma * eff * lumi

def compute_cls95_limit(B_obs, B_exp, S_template, edges, lo, hi, rng, ntrials=10000):
    """Compute σ_obs using the CLs method with toys."""
    s_vals = np.linspace(0, 10 * S_template.sum(), 100)
    cls_vals = []

    for s in s_vals:
        toys_sb = []
        toys_b = []

        for _ in range(ntrials):
            bkg = rng.poisson(B_exp)
            sb = rng.poisson(bkg + s)
            toys_sb.append(sb)
            toys_b.append(bkg)

        toys_sb = np.array(toys_sb)
        toys_b = np.array(toys_b)

        p_sb = np.count_nonzero(toys_sb >= B_obs) / ntrials
        p_b = np.count_nonzero(toys_b >= B_obs) / ntrials
        cls = p_sb / p_b if p_b > 0 else 1.0
        cls_vals.append(cls)

        if cls < 0.05:
            return s  # return first crossing point

    return s_vals[-1]  # fallback if no crossing
# 5. Plot helpers (unchanged except for logging)
# ----------------------------------------------------------------------
def plot_with_band(masses, values, lows, highs, ylabel, fname):
    plt.figure()
    plt.plot(masses, values, label="central", color="black")
    if lows is not None and highs is not None:
        plt.fill_between(masses, lows, highs, alpha=0.3, label="68% band")
    plt.xlabel("ALP Mass [GeV]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path("Plots").mkdir(exist_ok=True)
    plt.savefig(f"Plots/{fname}")
    plt.close()

def plot_z_vs_s95(zs, s95s):
    plt.figure()
    plt.scatter(zs, s95s)
    plt.xlabel("Expected Asimov Z")
    plt.ylabel("Expected s95 (yield)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/z_vs_s95.png")
    plt.close()


def plot_distribution(vals, med, p16, p84, path, case, sig_label, sigma_scale):
    """Single‑mode distribution plot (either S+B or B‑only)."""
    LOGGER.info("Saving plot → %s", path)
    finite = vals[np.isfinite(vals)]
    plt.figure(figsize=(7, 5))
    color = "deepskyblue" if case == "s_plus_b" else "grey"
    plt.hist(finite, bins=150, histtype="stepfilled", alpha=0.75, color=color)
    plt.axvline(med, ls="--", lw=1.2, label=f"median = {med:.2f}")
    plt.axvline(p16, color="k", ls=":", lw=1)
    plt.axvline(p84, color="k", ls=":", lw=1, label="68 % band")
    # xlabel = r"$Z = \dfrac{S}{\sqrt{B}}$" if case == "s_plus_b" else r"$\sqrt{B}$"
    xlabel = r"$Z = \sqrt{-2\log\lambda}$"
    plt.title(f"{case.replace('_', ' ')} for {sig_label} ({sigma_scale}σ fluct.)")
    plt.xlabel(f"{xlabel} (80 % ALP window; $m_{{low}}≥{MIN_MASS_EDGE}$ GeV)")
    plt.ylabel("Toy experiments")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()



def plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, path, sig_label, sigma_scale):
    import matplotlib.ticker as ticker

    # Clean values
    v_sb = np.array(vals_sb)[np.isfinite(vals_sb)]
    v_bo = np.array(vals_bo)[np.isfinite(vals_bo)]
    (med_sb, p16_sb, p84_sb), (med_bo, p16_bo, p84_bo) = stats_sb, stats_bo

    plt.figure(figsize=(8, 5))
    bins = np.histogram_bin_edges(np.concatenate([v_sb, v_bo]), bins=100)

    plt.hist(v_bo, bins=bins, alpha=0.6, color="gray", label="B only", edgecolor="black")
    plt.hist(v_sb, bins=bins, alpha=0.5, color="deepskyblue", label="S + B", edgecolor="black")
    # KDE Overlay (smooth curves)
    # sns.kdeplot(v_bo, bw_adjust=0.5, color="black", linestyle="--", linewidth=2.0, label="B only KDE")
    # sns.kdeplot(v_sb, bw_adjust=0.5, color="navy", linestyle="-", linewidth=2.0, label="S + B KDE")
    # Vertical lines for medians
    plt.axvline(med_bo, ls="--", lw=1.4, color="gray", label=f"Median B-only = {med_bo:.2f}")
    plt.axvline(med_sb, ls="--", lw=1.4, color="deepskyblue", label=f"Median S+B = {med_sb:.2f}")

    # Optional: add 68% interval shaded boxes
    plt.axvspan(p16_bo, p84_bo, color="gray", alpha=0.2)
    plt.axvspan(p16_sb, p84_sb, color="deepskyblue", alpha=0.2)

    # Labels and formatting
    plt.title(f"Toy Significance for {sig_label.replace('_', ' ')}")
    # plt.xlabel(r"$Z = \sqrt{-2\log\lambda}$")
    plt.xlabel(r"$Z$")
    plt.ylabel("Number of Toy Experiments")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

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
        coarse_edges = compute_coarse_binning(cnt_sig, edges, min_signal=0.001, merge_above=40.0)
        print(f"[{sig_key}] Using coarse binning: {coarse_edges}")
        bin_centers = (edges[:-1] + edges[1:]) / 2
        cnt_sig, _ = np.histogram(bin_centers, bins=coarse_edges, weights=cnt_sig)
        cnt_bkg, _ = np.histogram(bin_centers, bins=coarse_edges, weights=cnt_bkg)
        edges = coarse_edges
        # Map original fine bin edges to new indices in coarse binning
        lo = np.searchsorted(edges, lo_edge, side="left")
        hi = np.searchsorted(edges, hi_edge, side="right")

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
    plt.figure()
    plt.step(edges[:-1], cnt_sig, where="post", label="Signal")
    plt.step(edges[:-1], cnt_bkg, where="post", label="Background")
    plt.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")
    plt.legend(); plt.title(sig_key); plt.tight_layout();
    plt.yscale("log") 
    plt.savefig("some_plot.png")
    plt.close()

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
    # vals = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, "s_plus_b")
    vals = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, args.sigma, "s_plus_b",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
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

    while len(trials) < args.ntrials:
        bkg_shape = rng.poisson(cnt_bkg)
        samples = get_random_from_hist(edges, bkg_shape, rng, n_samples=bkg_shape.sum())
        toy_hist, _ = np.histogram(samples, bins=edges)
        b_toy = toy_hist[lo:hi].sum()

        if b_toy < MIN_B:
            rejected += 1
            continue  # skip this toy
        if args.use_cls:
            s95 = compute_cls95_limit(B_obs=b_toy, B_exp=B_exp, S_template=cnt_sig, edges=edges, lo=lo, hi=hi, rng=rng, ntrials=100)
        else:
            s95 = compute_s95_likelihood(0, b_toy)
        trials.append(s95)

    if rejected > 0:
        LOGGER.info(f"[{sig_key}] Skipped {rejected} trials with B < {MIN_B}")

    trials = np.array(trials)
    s95_med = np.percentile(trials, 50)
    p16, p84 = np.percentile(trials, [16, 84])
    p2p5, p97p5 = np.percentile(trials, [2.5, 97.5])
    s95_results[sig_key] = (s95_med, p16, p84, p2p5, p97p5)

    if eff > 0:
        sigma_excl     = s95_med / (eff * LUMINOSITY)
        sigma_excl_lo  = p16     / (eff * LUMINOSITY)
        sigma_excl_hi  = p84     / (eff * LUMINOSITY)
        sigma_excl_lo2  = p2p5     / (eff * LUMINOSITY)
        sigma_excl_hi2  = p97p5    / (eff * LUMINOSITY)        
    else:
        sigma_excl = sigma_excl_lo = sigma_excl_hi = float("nan")

    results[sig_key]["sigma_exclusion"]    = sigma_excl
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
        toy_mu95 = rng.poisson(cnt_bkg + sig_mu95)
        toy_large = rng.poisson(cnt_bkg + sig_large)

        plt.figure()
        plt.step(edges[:-1], cnt_bkg, where="post", label="Background", color="gray")
        plt.step(edges[:-1], sig_mu95, where="post", label="μ₉₅ × Signal", linestyle="--", color="blue")
        plt.step(edges[:-1], toy_mu95, where="post", label="Toy (B + μ₉₅ S)", color="black")
        plt.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")
        plt.title(f"{sig_key} — μ₉₅ Toy Overlay")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"toy_overlay_mu95_{sig_key}.png")
        plt.close()

        plt.figure()
        plt.step(edges[:-1], cnt_bkg, where="post", label="Background", color="gray")
        plt.step(edges[:-1], sig_large, where="post", label="20μ × Signal", linestyle="--", color="blue")
        plt.step(edges[:-1], toy_large, where="post", label="Toy (B + 20μ S)", color="black")
        plt.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")
        plt.title(f"{sig_key} — 20μ Toy Overlay")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"toy_overlay_mu20{sig_key}.png")
        plt.close()


    else:
        LOGGER.warning(f"[{sig_key}] No σ_nominal found — skipping μ₉₅ overlay")
    # Observed σ₉₅ exclusion (1 toy)
    # # -------------------------------
    MIN_B = 1  # global or local constant

    B_obs = rng.poisson(B_exp)  # Simulate "observed" event count in ROI
    if B_obs < MIN_B:
        LOGGER.warning(f"[{sig_key}] B_obs too small ({B_obs:.2f}), using fallback {MIN_B}")
        B_obs = MIN_B

    if args.use_cls:
        logging.info(f"[{sig_key}] Using CLs method for observed limit")
        s95_obs = compute_cls95_limit(
            B_obs, B_exp, cnt_sig, edges, lo, hi, rng, ntrials=1
        )
    else:
        s95_obs = compute_s95_likelihood(B_obs, B_exp)

    sigma_obs = s95_obs / (eff * LUMINOSITY)
    results[sig_key]["sigma_observed"] = sigma_obs
    logging.info(f"[Observed σ] {sig_key}: σ_obs_95% = {sigma_obs:.3f} nb (B_obs = {B_exp:.2f})")

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
        b_toy = rng.poisson(B_exp)
        if cnt_sig[lo:hi].sum() == 0 or b_toy == 0:
            continue
        mu_trial = 5 * math.sqrt(b_toy) / cnt_sig[lo:hi].sum()
        discovery_trials.append(mu_trial)

    discovery_trials = np.array(discovery_trials)
    mu_disc_median = np.percentile(discovery_trials, 50)
    mu_disc_lo1 = np.percentile(discovery_trials, 16)
    mu_disc_hi1 = np.percentile(discovery_trials, 84)
    mu_disc_lo2 = np.percentile(discovery_trials, 2.5)
    mu_disc_hi2 = np.percentile(discovery_trials, 97.5)

    results[sig_key]["mu_discovery_median"] = mu_disc_median
    results[sig_key]["mu_discovery_lo1"] = mu_disc_lo1
    results[sig_key]["mu_discovery_hi1"] = mu_disc_hi1
    results[sig_key]["mu_discovery_lo2"] = mu_disc_lo2
    results[sig_key]["mu_discovery_hi2"] = mu_disc_hi2

    # Also convert to σ (discovery threshold)
    sigma_disc = mu_disc_median / (eff * LUMINOSITY)
    sigma_disc_lo = mu_disc_lo1 / (eff * LUMINOSITY)
    sigma_disc_hi = mu_disc_hi1 / (eff * LUMINOSITY)
    sigma_disc_lo2 = mu_disc_lo2 / (eff * LUMINOSITY)
    sigma_disc_hi2 = mu_disc_hi2 / (eff * LUMINOSITY)

    results[sig_key]["sigma_discovery"] = sigma_disc
    results[sig_key]["sigma_discovery_lo"] = sigma_disc_lo
    results[sig_key]["sigma_discovery_hi"] = sigma_disc_hi
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
        # vals_sb = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, "s_plus_b")
        # vals_bo = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, "bkg_only")
        vals_sb = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, args.sigma, "s_plus_b",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
        vals_bo = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, args.sigma, "bkg_only",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
        stats_sb = (np.median(vals_sb), *np.percentile(vals_sb, [16, 84]))
        stats_bo = (np.median(vals_bo), *np.percentile(vals_bo, [16, 84]))

        out_dir = Path("Significance_dis") / REGION / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "overlay_80pc_distribution.png"
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, plot_path, sig_key, args.sigma)
        LOGGER.info("[OK] Overlay plot saved → %s", plot_path)
        

        # vals_sb = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, "s_plus_b")
        # vals_bo = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, "bkg_only")
        vals_sb = run_toys_likelihood_based(rng, sig_mu95, B_exp, args.ntrials, args.sigma, "s_scaled_plus_b",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
        vals_bo = run_toys_likelihood_based(rng, sig_mu95, B_exp, args.ntrials, args.sigma, "bkg_only",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
        stats_sb = (np.median(vals_sb), *np.percentile(vals_sb, [16, 84]))
        stats_bo = (np.median(vals_bo), *np.percentile(vals_bo, [16, 84]))

        out_dir = Path("Significance_dis") / REGION / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "overlay_80pc_distribution_scaled.png"
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, plot_path, sig_key, args.sigma)
        LOGGER.info("[OK] Overlay plot saved → %s", plot_path)
        

        # vals_sb = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, "s_plus_b")
        # vals_bo = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, "bkg_only")
        vals_sb = run_toys_likelihood_based(rng, sig_large, B_exp, args.ntrials, args.sigma, "s_scaled_plus_b",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
        vals_bo = run_toys_likelihood_based(rng, sig_mu95, B_exp, args.ntrials, args.sigma, "bkg_only",cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
        stats_sb = (np.median(vals_sb), *np.percentile(vals_sb, [16, 84]))
        stats_bo = (np.median(vals_bo), *np.percentile(vals_bo, [16, 84]))

        out_dir = Path("Significance_dis") / REGION / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "overlay_80pc_distribution_scaled20.png"
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, plot_path, sig_key, args.sigma)
        LOGGER.info("[OK] Overlay plot saved → %s", plot_path)
        


        sig_5sigma = mu_5sigma * cnt_sig
        toy_5sigma = rng.poisson(cnt_bkg + sig_5sigma)

        plt.figure()
        plt.step(edges[:-1], cnt_bkg, where="post", label="Background", color="gray")
        plt.step(edges[:-1], sig_5sigma, where="post", label="μ5σ × Signal", linestyle="--", color="green")
        plt.step(edges[:-1], toy_5sigma, where="post", label="Toy (B + μ5σ S)", color="black")
        plt.axvspan(edges[lo], edges[hi], color="orange", alpha=0.3, label="ROI")
        plt.title(f"{sig_key} — μ5σ Discovery Overlay")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"toy_discovery_mu5sigma_{sig_key}.png")
        plt.close()

    # Single‑case plots (either s_plus_b or bkg_only)
    mode = args.case
    vals = vals = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, args.sigma, "s_plus_b", cnt_bkg=cnt_bkg, edges=edges, lo=lo, hi=hi)
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
    plot_distribution(vals, med, p16, p84, plot_path, mode, sig_key, args.sigma)
    LOGGER.info("[OK] Plot saved → %s", plot_path)

    print(f"Non-finite Z vals (S+B): {np.sum(~np.isfinite(vals_sb))} / {len(vals_sb)}")
    print(f"Non-finite Z vals (B-only): {np.sum(~np.isfinite(vals_bo))} / {len(vals_bo)}")


def plot_asimov_summary(asimov_dict):
    import re
    import matplotlib.pyplot as plt
    masses = []
    z_vals = []
    for key, (_, _, z) in asimov_dict.items():
        match = re.match(r"alp_(\d+)GeV", key)
        if match:
            masses.append(int(match.group(1)))
            z_vals.append(z)
    sorted_pairs = sorted(zip(masses, z_vals))
    masses, z_vals = zip(*sorted_pairs)

    plt.figure()
    plt.plot(masses, z_vals, marker='o', color='blue', label='Asimov Z')
    plt.xlabel("ALP Mass [GeV]")
    plt.ylabel("Expected Asimov Z")
    plt.title("Expected Sensitivity vs ALP Mass")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("asimov_vs_mass.png")
    plt.close()
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

        # Plot: Z vs mass
        plt.figure()
        plt.plot(masses, zvals, "o-", color="blue", label="Asimov Z")
        plt.xlabel("ALP Mass [GeV]")
        plt.ylabel("Expected Asimov Z")
        plt.title("Expected Sensitivity vs ALP Mass")
        plt.legend()
        plt.tight_layout()
        plt.savefig("expected_asimov_z_vs_mass.png")
        plt.close()

        # Plot: s95 vs mass
        plt.figure()
        plt.plot(masses, s95vals, "o-", color="red", label="s95 (expected)")
        plt.xlabel("ALP Mass [GeV]")
        plt.ylabel("Expected Limit on Signal Yield (s95)")
        plt.title("Expected s95 Limit vs ALP Mass")
        plt.legend()
        plt.tight_layout()
        plt.savefig("expected_s95_vs_mass.png")
        plt.close()

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

        masses, s95_med, yerr_lo, yerr_hi = [], [], [], []
        for sig in sorted(s95_results.keys(), key=extract_mass_from_key):
            mass = extract_mass_from_key(sig)
            med, lo, hi, lo2, hi2 = s95_results[sig]
            masses.append(mass)
            s95_med.append(med)
            yerr_lo.append(med - lo)
            yerr_hi.append(hi - med)

        plt.errorbar(masses, s95_med, yerr=[yerr_lo, yerr_hi], fmt="o-", capsize=3, label="Median expected s95 ±1σ")
        plt.xlabel("ALP Mass [GeV]")
        plt.ylabel("s95 limit [events]")
        plt.title("Expected $s_{95}$ vs ALP Mass (Toy MC)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("expected_s95_vs_mass.png")
        plt.close()



        # -------------------------------
        # New: Exclusion cross section band
        # -------------------------------


        masses           = [extract_mass_from_key(k) for k in sig_keys]
        sigma_excl       = [results[k]["sigma_exclusion"] for k in sig_keys]
        sigma_excl_lo    = [results[k]["sigma_exclusion_lo"] for k in sig_keys]
        sigma_excl_hi    = [results[k]["sigma_exclusion_hi"] for k in sig_keys]
        sigma_excl_lo2   = [results[k]["sigma_exclusion_lo2"] for k in sig_keys]  # NEW
        sigma_excl_hi2   = [results[k]["sigma_exclusion_hi2"] for k in sig_keys]  # NEW
        sigma_observed   = [results[k]["sigma_observed"] for k in sig_keys]

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

        # Plot
        plt.figure(figsize=(7, 5))
        plt.plot(masses, sigma_excl, '-', color='black', label=r"Median σ$_{95}$ exclusion")

        # Plot ±2σ band first (underneath)
        plt.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                         color='yellow', alpha=0.3, label=r"±2σ band")

        # Plot ±1σ band on top
        plt.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                         color='green', alpha=0.4, label=r"±1σ band")

        plt.plot(masses, sigma_observed, 'k--', label=r"Observed σ$_{95}$")
        plt.xlabel("m$_{a}$ [GeV]")
        plt.ylabel("σ required for exclusion [nb]")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title("Required signal cross section vs ALP mass")
        plt.tight_layout()
        plt.savefig("required_sigma_exclusion_with_2sigma_band.png")

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
        sigma_observed   = [results[k]["sigma_observed"] for k in sig_keys]
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

        # Plot
        plt.figure(figsize=(7, 5))
        plt.plot(masses, excl_events, '-', color='black', label=r"Median $s_{95}$ (expected)")

        plt.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                         color='yellow', alpha=0.3, label=r"±2σ band")
        plt.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                         color='green', alpha=0.4, label=r"±1σ band")

        plt.plot(masses, obs_events, 'k--', label=r"Observed $s_{95}$")
        plt.xlabel("m$_{a}$ [GeV]")
        plt.ylabel("Expected signal events required for exclusion")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title("Required number of signal events for 95% CL exclusion vs ALP mass")
        plt.tight_layout()
        plt.savefig("required_events_exclusion_with_2sigma_band.png")
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

        plt.figure(figsize=(7, 5))
        plt.plot(masses, sigma_disc, '-', color='black', label=r"Median σ$_{disc}$ (Z=2)")
        plt.plot(masses, sigma_5sigma, '--', color='red', label=r"Z = 5 (5σ)")
        # Plot ±2σ band first
        plt.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                         color='yellow', alpha=0.3, label=r"±2σ band")

        # Plot ±1σ band on top
        plt.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                         color='green', alpha=0.4, label=r"±1σ band")

        plt.xlabel("m$_{a}$ [GeV]")
        plt.ylabel("σ required for discovery (Z=2) [nb]")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title("Required cross section for Z = 2 discovery vs ALP mass")
        plt.tight_layout()
        plt.savefig("required_sigma_discovery_with_2sigma_band.png")


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

        # Plot
        plt.figure(figsize=(7, 5))
        plt.plot(masses, disc_events, '-', color='black', label=r"Median $s_{disc}$ (Z=2)")
        plt.plot(masses, events_5sigma, '--', color='red', label=r"Z = 5 (5σ)")

        plt.fill_between(masses, band_lo_2sigma, band_hi_2sigma,
                         color='yellow', alpha=0.3, label=r"±2σ band")
        plt.fill_between(masses, band_lo_1sigma, band_hi_1sigma,
                         color='green', alpha=0.4, label=r"±1σ band")

        plt.xlabel("m$_{a}$ [GeV]")
        plt.ylabel("Signal events required for discovery")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title("Signal events required for Z = 2 (discovery) vs ALP mass")
        plt.tight_layout()
        plt.savefig("required_events_discovery_with_2sigma_band.png")



        # ----------------------------------------
        # Final: Print clean LIMIT SUMMARY per signal
        # ----------------------------------------
        print("\n===== ALL LIMIT SUMMARIES =====")
        for sig_key in sorted(results):
            print_limit_summary(sig_key, results[sig_key])
if __name__ == "__main__":
    main()
