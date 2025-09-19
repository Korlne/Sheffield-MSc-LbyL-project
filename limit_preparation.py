#!/usr/bin/env python3
"""
significance_analysis_root.py
--------------------------------------
Toy studies *inside the 80% ALP mass window* **with Gaussian-fluctuated background** for the new histogram-based pickle format - *acoplanarity region fixed to SR*.

This **debug-enhanced** version of the original script adds granular logging so you can inspect the internal flow, intermediate variables and ROI selection logic. Activate verbose output with ``--verbose`` (or ``-v``) on the command line.

Pickle layout expected (new format):
    all_histograms[sample] = {
        'nominal': {
            'h_ZMassZoom':   {'bin_edges': [...], 'counts': [...]},
            'h_ZMassFine':   {...},
            ...
        },
        'systematics': {
            'EG_SCALE_ALL__1up':     { <same histogram dicts> },
            'EG_SCALE_ALL__1down':   { ... },
            'EG_RESOLUTION_ALL__1up':{ ... },
            'EG_RESOLUTION_ALL__1down':{ ... }
        },
        'eff_sr': <float>   # selection efficiency (only for ALP & lbyl)
    }

ALP samples (keys beginning with ``alp_``) are treated as **signal**. The union of *lbyl, yy2ee* and *cep* forms the **background** by default.

Example (single ALP mass in verbose mode):
    python3 significance_analysis_root.py --signal alp_20GeV -v

Example (all ALP masses, overlay plot), processes each ALP sample automatically:
    python3 significance_analysis_root.py --signal all --case overlay
"""
# ----------------------------------------------------------------------
# 0. Imports & constants
# ----------------------------------------------------------------------
import argparse
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d  # for smoothing the band only
import matplotlib.pyplot as plt
import numpy as np

# Configure a module-level logger
LOGGER = logging.getLogger(__name__)

# Lowest allowed edge of the ROI (Region-of-Interest)
MIN_MASS_EDGE = 5.0  # GeV

# Default histogram & background configuration
DEFAULT_HIST_KEY = "h_ZMassFine"
DEFAULT_BKG = ["lbyl", "yy2ee", "cep"]

# Fixed acoplanarity region
REGION = "sr"

# Directory path holding the pickled histograms for SR
PKL_PATH = Path("/home/jtong/lbyl/bkg/bkg_alp_sr_pickle.pkl")

# ----------------------------------------------------------------------
# 1. CLI parsing (SR only)
# ----------------------------------------------------------------------
def parse_cli():
    """Parse command-line options relevant for the SR-only analysis."""
    p = argparse.ArgumentParser(
        description=(
            "Toy study inside the 80% ALP window (Gaussian-fluctuated background) "
            "for the histogram-based pickle format – SR region only."
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
    # Histogram to analyze
    p.add_argument(
        "--hist",
        default=DEFAULT_HIST_KEY,
        help=f"Histogram name to analyze (default: {DEFAULT_HIST_KEY})",
    )
    # Toy-MC configuration
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

# Global constants for normalization
LUMINOSITY = 1.63  # in nb^-1

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
    """Extract (edges, counts) arrays for a given sample and histogram key from the pickle.
    Automatically handles new pickle format where histograms are under the 'nominal' sub-dictionary.
    """
    LOGGER.debug("Fetching histogram '%s/%s'", sample, hist_key)
    # Determine if the sample has 'nominal' sub-dictionary (new format)
    if isinstance(pkl.get(sample), dict) and "nominal" in pkl[sample]:
        hist_group = pkl[sample]["nominal"]
    else:
        hist_group = pkl.get(sample, {})
    try:
        h = hist_group[hist_key]
        edges = np.asarray(h["bin_edges"], dtype=float)
        counts = np.asarray(h["counts"], dtype=float)
    except KeyError:
        LOGGER.error("Missing '%s/%s' in pickle", sample, hist_key)
        sys.exit(f"[FATAL] Missing '{sample}/{hist_key}' in pickle")
    LOGGER.debug("Histogram '%s/%s' fetched (bins=%d)", sample, hist_key, len(counts))
    return edges, counts

# ----------------------------------------------------------------------
# 3. ROI (Region-of-Interest) utilities
# ----------------------------------------------------------------------
def cumulative(cnt):
    """Cumulative sum with a leading zero – useful for fast integrals."""
    return np.concatenate(([0.0], np.cumsum(cnt)))

def integral(cum, lo, hi):
    """Integral of *cnt* between bin indices [lo, hi) using its cumulative array."""
    return cum[hi] - cum[lo]

def tight_window(cnt_sig, edges, i_cen, frac=0.80):
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

def enlarge_for_bkg(cnt_bkg, edges, lo, hi):
    """Ensure at least one background event inside the ROI (expands ROI if needed)."""
    bkg_cum = cumulative(cnt_bkg)
    if integral(bkg_cum, lo, hi) > 0.0:
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
        if integral(bkg_cum, left, right) > 0.0:
            LOGGER.debug("Enlarged ROI to include background: [%d, %d) bins", left, right)
            return left, right
    sys.exit("[FATAL] Could not build an 80% ALP window with non-zero background.")

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
        sys.exit("[FATAL] No valid ROI satisfying 80%/bkg>0 above 5 GeV")
    LOGGER.info("Best ROI bins=[%d,%d) → expected Z=%.3f", best_lo, best_hi, best_z)
    return best_lo, best_hi, best_z

def find_full_roi(cnt_sig, cnt_bkg, edges):
    """Return ROI spanning the full histogram range."""
    return 0, len(cnt_sig), None

# ----------------------------------------------------------------------
# 4. Toy-MC utilities
# ----------------------------------------------------------------------
def compute_asimov_significance(signal, background):
    """Compute the Asimov significance using the formula:
    \( Z = \sqrt{2\left[(s+b) \ln\left(1 + \frac{s}{b}\right) - s\right]} \).
    Returns 0.0 if either signal or background is non-positive.
    """
    if background <= 0 or signal <= 0:
        return 0.0
    return np.sqrt(2 * ((signal + background) * np.log(1 + signal / background) - signal))

def draw_bg(rng, b_exp, sigma_scale):
    """Generate a Gaussian-fluctuated background yield (≥0)."""
    std = math.sqrt(b_exp) * sigma_scale
    return max(rng.normal(b_exp, std), 0.0)

def run_toys(rng, S_fix, B_exp, ntrials, sigma_scale=1.0, mode="s_plus_b"):
    """Run toy experiments computing significance Z = S/√B for each trial.
    If mode == 's_plus_b', both signal (S) and background (B) yields fluctuate (Poisson) around S_fix and B_exp.
    If mode == 'bkg_only', only background fluctuates (signal is fixed at S_fix without fluctuation).
    Returns an array of Z values for each toy trial. Z = 0 for trials where B_fluct = 0 (to avoid division by zero).
    """
    vals = np.zeros(ntrials)
    for i in range(ntrials):
        B_fluct = rng.poisson(B_exp * sigma_scale)
        if mode == "s_plus_b":
            S_fluct = rng.poisson(S_fix * sigma_scale)
        else:
            S_fluct = S_fix  # keep signal fixed in background-only mode
        if B_fluct > 0:
            vals[i] = S_fluct / math.sqrt(B_fluct)
        else:
            vals[i] = 0.0
    return vals

def run_toys_likelihood_based(rng, S_fix, B_exp, n_trials, sigma_scale=1.0, mode="s_plus_b"):
    """
    Run toy experiments using a log-likelihood ratio test statistic for significance.

    Args:
        rng: NumPy random generator (e.g. np.random.default_rng()).
        S_fix: Expected signal yield (in ROI) for mu=1 hypothesis.
        B_exp: Expected background yield (in ROI).
        n_trials: Number of toy experiments to simulate.
        sigma_scale: Scale factor for background fluctuation (defaults to 1.0 for nominal fluctuations).
        mode: Either 's_plus_b' or 'bkg_only'. Determines if signal is injected or not.

    Returns:
        Numpy array of test statistic values (approximately Z = sqrt{-2 ln λ}) for each toy trial.
    """
    vals = np.zeros(n_trials)
    for i in range(n_trials):
        # Fluctuate background count
        B_fluct = rng.poisson(B_exp * sigma_scale)
        # Fluctuate signal count if in S+B mode
        if mode == "s_plus_b":
            S_fluct = rng.poisson(S_fix)
        elif mode == "bkg_only":
            S_fluct = 0
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # Observed total events in toy
        N_obs = S_fluct + B_fluct
        # Define null (background-only) and alt (signal+background) hypotheses
        mu_null = 0.0
        B_null = B_exp
        expected_null = mu_null * S_fix + B_null
        mu_alt = 1.0
        expected_alt = mu_alt * S_fix + B_exp
        # Poisson log-likelihoods (constant terms dropped)
        if expected_alt > 0 and expected_null > 0 and N_obs > 0:
            try:
                logL_alt = N_obs * math.log(expected_alt) - expected_alt
                logL_null = N_obs * math.log(expected_null) - expected_null
                q_mu = -2 * (logL_null - logL_alt)
                # Signed sqrt: positive if N_obs > expected_null (excess events), negative if deficit
                vals[i] = math.copysign(math.sqrt(abs(q_mu)), N_obs - expected_null)
            except ValueError:
                vals[i] = 0.0  # fallback if log computation fails
        else:
            vals[i] = 0.0
    return vals

def extract_mass_from_key(key):
    """Extract the numeric mass value (in GeV) from a sample key like 'alp_20GeV'."""
    try:
        return float(key.split("_")[1].replace("GeV", "").replace("p", "."))
    except Exception:
        return -1.0

def compute_s95_likelihood(s, b, cl=0.95):
    """Compute the 95% CL upper limit on signal (s95) using Poisson statistics (CLs method)."""
    from scipy.stats import poisson
    limit = 0
    # Increment the candidate signal count until the cumulative probability exceeds the confidence level
    while poisson.cdf(limit, b + s) < cl:
        limit += 1
    return limit

def get_required_sigma(target_Z, eff, lumi, bkg, tol=1e-3, max_sigma=1e4):
    """
    Scale cross section σ until s = σ * eff * lumi yields Z ≥ target_Z (using s/√b as estimate).
    Uses binary search to find the minimal σ that achieves target_Z.
    """
    low, high = 0.0, max_sigma
    for _ in range(50):  # binary search loop
        mid = 0.5 * (low + high)
        s = mid * eff * lumi
        Z = s / np.sqrt(bkg) if bkg > 0 else 0.0
        if abs(Z - target_Z) < tol:
            return mid
        if Z < target_Z:
            low = mid
        else:
            high = mid
    return mid

# ----------------------------------------------------------------------
# 5. Plot helpers
# ----------------------------------------------------------------------
def plot_with_band(masses, values, lows, highs, ylabel, fname):
    """Plot an array of values vs masses with an optional ±1σ band, then save to file (PNG)."""
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
    """Scatter plot of Asimov Z vs required signal yield s95, saved to file."""
    plt.figure()
    plt.scatter(zs, s95s)
    plt.xlabel("Expected Asimov Z")
    plt.ylabel("Expected s95 (yield)")
    plt.grid(True)
    plt.tight_layout()
    Path("Plots").mkdir(exist_ok=True)
    plt.savefig("Plots/z_vs_s95.png")
    plt.close()

def plot_distribution(vals, med, p16, p84, path, case, sig_label, sigma_scale):
    """Plot a single distribution (either S+B or B-only) of toy experiments and save it as a histogram."""
    LOGGER.info("Saving plot → %s", path)
    finite = np.asarray(vals)[np.isfinite(vals)]
    plt.figure(figsize=(7, 5))
    color = "deepskyblue" if case == "s_plus_b" else "grey"
    plt.hist(finite, bins=150, histtype="stepfilled", alpha=0.75, color=color)
    plt.axvline(med, ls="--", lw=1.2, label=f"median = {med:.2f}")
    plt.axvline(p16, color="k", ls=":", lw=1)
    plt.axvline(p84, color="k", ls=":", lw=1, label="68 % band")
    xlabel = r"$Z = \sqrt{-2\log\lambda}$"
    plt.title(f"{case.replace('_', ' ')} for {sig_label} ({sigma_scale}σ fluct.)")
    plt.xlabel(f"{xlabel} (80% ALP window; $m_{{low}}≥{MIN_MASS_EDGE}$ GeV)")
    plt.ylabel("Toy experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()

def plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, path, sig_label, sigma_scale):
    """Overlay S+B and B-only toy distributions on the same plot (using KDE for smooth curves)."""
    # Clean finite values
    v_sb = np.array(vals_sb)[np.isfinite(vals_sb)]
    v_bo = np.array(vals_bo)[np.isfinite(vals_bo)]
    (med_sb, p16_sb, p84_sb), (med_bo, p16_bo, p84_bo) = stats_sb, stats_bo
    plt.figure(figsize=(8, 5))
    # Use a common bin range for both distributions
    bins = np.histogram_bin_edges(np.concatenate([v_sb, v_bo]), bins=200)
    # Plot smooth KDE curves for each distribution
    xs = np.linspace(min(v_bo.min(), v_sb.min()), max(v_bo.max(), v_sb.max()), 1000)
    kde_bo = gaussian_kde(v_bo, bw_method=0.5)
    kde_sb = gaussian_kde(v_sb, bw_method=0.5)
    plt.plot(xs, kde_bo(xs), "--", color="black", linewidth=2.0, label="B only KDE")
    plt.plot(xs, kde_sb(xs),  "-", color="navy",  linewidth=2.0, label="S + B KDE")
    # Mark medians with vertical lines
    plt.axvline(med_bo, ls="--", lw=1.4, color="gray", label=f"Median B-only = {med_bo:.2f}")
    plt.axvline(med_sb, ls="--", lw=1.4, color="deepskyblue", label=f"Median S+B = {med_sb:.2f}")
    # Shade 68% interval regions
    plt.axvspan(p16_bo, p84_bo, color="gray", alpha=0.2)
    plt.axvspan(p16_sb, p84_sb, color="deepskyblue", alpha=0.2)
    # Labels and formatting
    plt.title(f"Toy Significance for {sig_label.replace('_', ' ')}")
    plt.xlabel(r"$Z = \sqrt{-2\log\lambda}$")
    plt.ylabel("Number of Toy Experiments")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ----------------------------------------------------------------------
# 6. Core processing function for a single signal (SR only)
# ----------------------------------------------------------------------
def process(args, sig_key):
    """Process a single ALP signal sample in the SR region.
    Loads histograms from the pickle, finds the optimal 80% containment ROI for signal (ensuring ≥1 background event),
    computes expected signal (S_fix) and background (B_exp) yields in the ROI, then calculates the Asimov significance.
    Performs toy Monte Carlo experiments to estimate the distribution of the test statistic and derive the median s95 (events) and corresponding σ_95 (cross section) for exclusion.
    Saves distribution plots if requested and accumulates results for summary plots.
    """
    LOGGER.debug("Processing signal '%s'", sig_key)
    rng = np.random.default_rng(args.seed)
    # 6a. Load pickled histograms and get signal counts
    pkl = load_pickle(PKL_PATH)
    edges, cnt_sig = fetch_histogram(pkl, sig_key, args.hist)
    # Skip processing if signal histogram is empty
    if cnt_sig.sum() == 0:
        LOGGER.warning("Skipping %s: empty signal histogram", sig_key)
        return
    # Sum up the background histograms from all specified background samples
    cnt_bkg = np.zeros_like(cnt_sig, dtype=float)
    for b in args.bkg:
        _, cnt_tmp = fetch_histogram(pkl, b, args.hist)
        cnt_bkg += cnt_tmp
    LOGGER.debug("Background summed across %d samples: %s", len(args.bkg), args.bkg)
    # 6b. Determine the Region of Interest (ROI) in mass
    if args.roi == "full":
        lo, hi, z_exp = find_full_roi(cnt_sig, cnt_bkg, edges)
        roi_desc = "full range"
    else:
        lo, hi, z_exp = find_best_roi(cnt_sig, cnt_bkg, edges)
        roi_desc = f"{edges[lo]:.1f}–{edges[hi]:.1f} GeV"
    lo_edge, hi_edge = edges[lo], edges[hi]
    if z_exp is not None:
        LOGGER.info("[SR/%s] ROI: [%.2f, %.2f] GeV (expected Z=%.3f)", sig_key, lo_edge, hi_edge, z_exp)
    else:
        LOGGER.info("[SR/%s] ROI: [%.2f, %.2f] GeV (expected Z=unknown)", sig_key, lo_edge, hi_edge)
    # Fixed S and expected B counts inside the chosen ROI
    S_fix = cnt_sig[lo:hi].sum()
    B_exp = cnt_bkg[lo:hi].sum()
    LOGGER.debug("[ROI summary] bins=[%d,%d)  S=%.3f  B=%.3f  sqrt(B)=%.3f", lo, hi, S_fix, B_exp, math.sqrt(B_exp) if B_exp else 0.0)
    if S_fix == 0 or B_exp == 0:
        LOGGER.warning("Skipping %s: empty S or B in ROI (S=%.3f, B=%.3f)", sig_key, S_fix, B_exp)
        return
    # Retrieve selection efficiency for this signal (or use default 0.35 if not provided)
    eff = pkl[sig_key].get("eff_sr", 0.35) if isinstance(pkl[sig_key], dict) else 0.35
    LOGGER.info("[eff_sr] %s → ε = %.3f", sig_key, eff)
    # 6c. Compute Asimov significance and run toy experiments
    z_asimov = compute_asimov_significance(S_fix, B_exp)
    LOGGER.info("[Asimov Z] %s: s=%.3f, b=%.3f, Z=%.2f", sig_key, S_fix, B_exp, z_asimov)
    asimov_results[sig_key] = (S_fix, B_exp, z_asimov)
    # Record in results dictionary
    if sig_key not in results:
        results[sig_key] = {}
    results[sig_key]["z"] = z_asimov
    # Compute distribution of test statistic under S+B via toy MC
    vals = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, args.sigma, mode="s_plus_b")
    # Determine the 95th percentile and 68% (16-84) range of the test statistic distribution
    s95 = np.percentile(vals, 95)
    s68lo = np.percentile(vals, 16)
    s68hi = np.percentile(vals, 84)
    results[sig_key]["s95"] = s95
    results[sig_key]["s95_lo"] = s68lo
    results[sig_key]["s95_hi"] = s68hi
    # Convert the 95th percentile test statistic to an equivalent cross section scale (not used further)
    results[sig_key]["sigma95"] = s95 / (eff * LUMINOSITY) if eff > 0 else float('inf')
    # Toy trials to estimate required signal yield for exclusion (CL=95%)
    trials = []
    for _ in range(args.ntrials):
        b_toy = rng.poisson(B_exp)
        if b_toy == 0:
            s95_i = 1.0  # if no background events, 1 signal event would already exclude at 95% CL
        else:
            s95_i = compute_s95_likelihood(0, b_toy)
        trials.append(s95_i)
    trials = np.array(trials)
    s95_med = np.median(trials)
    p16, p84 = np.percentile(trials, [16, 84])
    s95_results[sig_key] = (s95_med, p16, p84)
    # Convert required signal yield to required cross section for exclusion
    if eff > 0:
        sigma_excl = s95_med / (eff * LUMINOSITY)
        sigma_excl_lo = p16 / (eff * LUMINOSITY)
        sigma_excl_hi = p84 / (eff * LUMINOSITY)
    else:
        sigma_excl = sigma_excl_lo = sigma_excl_hi = float("nan")
    results[sig_key]["sigma_exclusion"] = sigma_excl
    results[sig_key]["sigma_exclusion_lo"] = sigma_excl_lo
    results[sig_key]["sigma_exclusion_hi"] = sigma_excl_hi
    LOGGER.info(
        "[Exclusion σ] %s: σ_95%% = %.3f nb [−1σ=%.3f, +1σ=%.3f]",
        sig_key, sigma_excl, sigma_excl_lo, sigma_excl_hi
    )
    # If overlay case, produce combined S+B vs B-only distribution plot and return (skip further modes)
    if args.case == "overlay":
        vals_sb = vals  # S+B distribution already computed
        vals_bo = run_toys_likelihood_based(rng, S_fix, B_exp, args.ntrials, args.sigma, mode="bkg_only")
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
        stats_sb = (np.median(vals_sb), np.percentile(vals_sb, 16), np.percentile(vals_sb, 84))
        stats_bo = (np.median(vals_bo), np.percentile(vals_bo, 16), np.percentile(vals_bo, 84))
        out_dir = Path("Significance_dis") / REGION / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "overlay_80pc_distribution.png"
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, plot_path, sig_key, args.sigma)
        LOGGER.info("[OK] Overlay plot saved → %s", plot_path)
        return
    # If single-case requested, plot that distribution
    mode = args.case  # 's_plus_b' or 'bkg_only'
    vals_mode = run_toys(rng, S_fix, B_exp, args.ntrials, args.sigma, mode)
    med_val = np.median(vals_mode)
    p16_val, p84_val = np.percentile(vals_mode, [16, 84])
    p025_val, p975_val = np.percentile(vals_mode, [2.5, 97.5]) if len(vals_mode) > 0 else (0, 0)
    if mode == "s_plus_b":
        LOGGER.info(
            "Trials: %d  Median Z: %.3f  68%%: [%.3f, %.3f]  95%%: [%.3f, %.3f]",
            args.ntrials, med_val, p16_val, p84_val, p025_val, p975_val
        )
    else:
        LOGGER.info(
            "Trials: %d  Median √B: %.3f  68%%: [%.3f, %.3f]  95%%: [%.3f, %.3f]",
            args.ntrials, med_val, p16_val, p84_val, p025_val, p975_val
        )
    out_dir = Path("Significance_dis") / REGION / sig_key
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "alp" if mode == "s_plus_b" else "bkg_only"
    plot_path = out_dir / f"{tag}_80pc_distribution.png"
    plot_distribution(vals_mode, med_val, p16_val, p84_val, plot_path, mode, sig_key, args.sigma)
    LOGGER.info("[OK] Plot saved → %s", plot_path)

# ----------------------------------------------------------------------
# 7. Main execution loop (SR only)
# ----------------------------------------------------------------------
def main():
    args = parse_cli()
    # Configure logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.verbose:
        # Reduce verbosity of external libraries when debug is on
        for noisy in ("matplotlib", "matplotlib.font_manager", "PIL"):
            logging.getLogger(noisy).setLevel(logging.INFO)
    LOGGER.debug("Command-line arguments: %s", args)
    # Ensure the pickle file exists
    if not PKL_PATH.is_file():
        sys.exit(f"[FATAL] Expected pickle '{PKL_PATH}' not found.")
    # Load pickle to discover available sample keys
    pkl_preview = load_pickle(PKL_PATH)
    # Determine which signal samples to process
    if args.signal.lower() in {"all", "alp_all"}:
        sigs = sorted([k for k in pkl_preview if str(k).startswith("alp_")])
        if not sigs:
            sys.exit(f"[FATAL] No ALP samples in '{PKL_PATH}'")
    else:
        sigs = [args.signal]
    LOGGER.info("Signals to process: %s", ", ".join(sigs))
    # Process each requested ALP mass
    for sig in sigs:
        process(args, sig)
    # Generate summary plots if we ran the full overlay case
    if args.case == "overlay" and asimov_results:
        # Prepare sorted arrays of masses for signals processed
        sig_keys = sorted(asimov_results.keys(), key=extract_mass_from_key)
        masses = [extract_mass_from_key(k) for k in sig_keys]
        # 3. Plot: expected_asimov_z_vs_mass.png (Expected Asimov significance vs ALP mass)
        # Y-axis: Expected Asimov Z-significance
        # This uses the Asimov dataset formula (no fluctuations) to estimate significance:
        #   Z = sqrt(2 * [ (s+b) * ln(1 + s/b) - s ])
        # It shows the expected significance for each ALP mass given the predicted signal (S_fix) and background (B_exp) in the ROI.
        # Useful for quick sensitivity estimates, though it can underestimate the variance when counts are low.
        zvals = [asimov_results[k][2] for k in sig_keys]
        plot_with_band(masses, zvals, None, None, "Expected Asimov Z", "expected_asimov_z_vs_mass.png")
        # 1. Plot: expected_s95_vs_mass.png (Median required signal events vs ALP mass)
        # Y-axis: Median s95 [number of signal events]
        # This is a toy-MC estimate of how many signal events are needed to exclude the background at 95% CL.
        # For each mass, we fluctuate the background and compute the 95% CL limit on signal (s95) for many trials,
        # then take the median and ±1σ (16th–84th percentile) range. This is model-independent (just event counts).
        masses_s95, med_s95, err_lo, err_hi = [], [], [], []
        for sig in sig_keys:
            mass = extract_mass_from_key(sig)
            med, lo, hi = s95_results[sig]
            masses_s95.append(mass)
            med_s95.append(med)
            err_lo.append(med - lo)
            err_hi.append(hi - med)
        plt.figure()
        plt.errorbar(masses_s95, med_s95, yerr=[err_lo, err_hi], fmt="o-", capsize=3, label="Median s95 ±1σ")
        plt.xlabel("ALP Mass [GeV]")
        plt.ylabel("s95 limit [events]")
        plt.title("Expected $s_{95}$ vs ALP Mass (Toy MC)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        Path("Plots").mkdir(exist_ok=True)
        plt.savefig("Plots/expected_s95_vs_mass.png")
        plt.close()
        # 2. Plot: required_sigma_exclusion_with_band.png (Required signal cross section vs ALP mass)
        # Y-axis: Required signal cross section σ [nb]
        # This converts the median required signal yield (s95) into a cross section limit using σ_95 = s95 / (eff × L).
        # It shows the median σ required for exclusion at 95% CL for each mass, with a ±1σ uncertainty band.
        sigma_excl = [results[k]["sigma_exclusion"] for k in sig_keys]
        sigma_excl_lo = [results[k]["sigma_exclusion_lo"] for k in sig_keys]
        sigma_excl_hi = [results[k]["sigma_exclusion_hi"] for k in sig_keys]
        # Compute asymmetric error band
        yerr_lo = [max(0, mid - lo) for mid, lo in zip(sigma_excl, sigma_excl_lo)]
        yerr_hi = [max(0, hi - mid) for mid, hi in zip(sigma_excl, sigma_excl_hi)]
        # Smooth the band edges for a cleaner look
        sigma_excl = np.array(sigma_excl)
        band_lo_smooth = gaussian_filter1d(sigma_excl - yerr_lo, sigma=1.0)
        band_hi_smooth = gaussian_filter1d(sigma_excl + yerr_hi, sigma=1.0)
        plt.figure(figsize=(7, 5))
        plt.plot(masses, sigma_excl, 'o-', color='C0', label=r"Median σ$_{95}$ exclusion")
        plt.fill_between(masses, band_lo_smooth, band_hi_smooth, color='C0', alpha=0.3, label=r"±1σ band")
        plt.xlabel("m$_{a}$ [GeV]")
        plt.ylabel("σ required for exclusion [nb]")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title("Required signal cross section vs ALP mass")
        plt.tight_layout()
        Path("Plots").mkdir(exist_ok=True)
        plt.savefig("Plots/required_sigma_exclusion_with_band.png")
        plt.close()
        # 4. Plot: z_vs_s95.png (Expected significance vs required signal yield)
        # Scatter plot showing Asimov significance (Z) versus the median required signal yield (s95) for each mass point.
        # Illustrates the inverse relationship: higher required event count corresponds to lower significance.
        zs = [asimov_results[k][2] for k in sig_keys]
        s95_yields = [s95_results[k][0] for k in sig_keys]
        plot_z_vs_s95(zs, s95_yields)
        LOGGER.info("[✓] Summary plots saved in 'Plots/' directory.")

if __name__ == "__main__":
    main()
