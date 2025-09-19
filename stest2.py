#!/usr/bin/env python3
"""
significance_analysis_root.py
--------------------------------------
Toy studies *inside the 80% ALP window* **with Poisson-fluctuated background** 
for the new histogram-based pickle format - *acoplanarity region fixed to SR*.

This reorganized version structures the workflow into smaller functions for clarity. 
It processes the updated pickle structure where each sample entry includes "nominal", 
"systematics", and "eff_sr" fields.
The pickle have the structure

    all_histograms[sample] = {
        "nominal": {
            "h_ZMassZoom":   {"bin_edges": [...], "counts": [...]},
            "h_ZMassFine":   {...},
            ⋯
        },
        "systematics": {
            "EG_SCALE_ALL__1up"     : { <same hist dicts> },
            "EG_SCALE_ALL__1down"   : { … },
            "EG_RESOLUTION__ALL_1up": { … },
            "EG_RESOLUTION__ALL_1down": { … },
        },
        "eff_sr": <float>   # only for ALP & lbyl
    }


Example (verbose mode, single ALP mass):
    python3 significance_analysis_root.py --signal alp_20GeV -v
Example (overlay plot for all ALPs):
    python3 significance_analysis_root.py --signal all --case overlay
"""
# ----------------------------------------------------------------------
# 0. Imports & Constants
# ----------------------------------------------------------------------
import argparse
import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

LOGGER = logging.getLogger(__name__)

LUMINOSITY = 1.63  # in nb^-1
MIN_MASS_EDGE = 5.0  # GeV
DEFAULT_BKG = ["lbyl", "yy2ee", "cep"]
DEFAULT_HIST_KEY = "h_ZMassFine"

PKL_PATH = Path("/home/jtong/lbyl/bkg/bkg_alp_sr_pickle.pkl")

# Nominal ALP production cross sections (nb) for each mass (for reference)
alp_sigma_nb = {
    4: 7.967330e3, 5: 6.953744e3, 6: 6.044791e3, 7: 5.300250e3,
    8: 4.670220e3, 9: 4.154600e3, 10: 3.709976e3, 12: 3.016039e3,
    14: 2.499097e3, 15: 2.285133e3, 16: 2.093761e3, 18: 1.782345e3,
    20: 1.526278e3, 30: 7.779030e2, 40: 4.368360e2, 50: 2.600118e2,
    60: 1.604056e2, 70: 1.016849e2, 80: 6.546058e1, 90: 4.280824e1,
    100: 2.824225e1,
}

# Storage for results
asimov_results = {}
results = {}
s95_results = {}

# ----------------------------------------------------------------------
# 1. CLI Parsing (SR only)
# ----------------------------------------------------------------------
def parse_cli():
    """Parse command-line options for significance analysis in SR."""
    parser = argparse.ArgumentParser(
        description="Toy study inside the 80% ALP window (Gaussian-fluctuated background) – SR region only."
    )
    parser.add_argument(
        "--signal", default="alp_20GeV",
        help="Signal key (e.g. 'alp_20GeV') or 'all' to process every ALP mass"
    )
    parser.add_argument(
        "--bkg", nargs="+", default=DEFAULT_BKG,
        help=f"Background sample keys (default: {' '.join(DEFAULT_BKG)})"
    )
    parser.add_argument(
        "--hist", default=DEFAULT_HIST_KEY,
        help=f"Histogram name to analyze (default: {DEFAULT_HIST_KEY})"
    )
    parser.add_argument("--ntrials", type=int, default=10000, help="Number of toy experiments")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian width scale (σ factor)")
    parser.add_argument(
        "--sigscale", type=float, default=1.0,
        help="Scale factor applied to the signal yield when computing the Asimov significance (default: 1.0)",
    )
    parser.add_argument(
        "--roi", choices=["best", "full"], default="best",
        help="ROI selection: 'best' (max Z) or 'full' spectrum"
    )
    parser.add_argument(
        "--case", choices=["overlay", "s_plus_b", "bkg_only"], default="overlay",
        help="'overlay' (both S+B and B-only), 's_plus_b' or 'bkg_only'"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug output")
    parser.add_argument("--coarse-binning", default=True, action="store_true",
                        help="Enable coarse optimized binning based on signal")
    parser.add_argument("--profile-systematics", action="store_true",
                        help="Enable shape profiling using Gaussian systematics pulls")
    parser.add_argument("--use-cls", action="store_true",
                        help="Use CLs-based exclusion instead of Z-based")
    return parser.parse_args()

# ----------------------------------------------------------------------
# 2. Histogram Helpers
# ----------------------------------------------------------------------
def load_pickle(path):
    """Load pickled histogram data or exit on failure."""
    LOGGER.debug("Loading pickle: %s", path)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        LOGGER.debug("Pickle loaded - samples: %s", list(data.keys())[:5])
        return data
    except Exception as e:
        LOGGER.exception("Cannot read pickle '%s'", path)
        sys.exit(f"[FATAL] Cannot read pickle '{path}': {e}")

def get_histogram(pkl, sample, hist_key, profile=False, rng=None):
    """
    Get histogram bin edges and counts for a sample.
    If profile=True, incorporate shape systematics via Gaussian pulls.
    """
    LOGGER.debug("Fetching histogram '%s/%s'", sample, hist_key)
    edges = np.array(pkl[sample]["nominal"][hist_key]["bin_edges"], dtype=float)
    counts = np.array(pkl[sample]["nominal"][hist_key]["counts"], dtype=float)
    if profile:
        shape = counts.copy()
        systs = pkl[sample].get("systematics", {})
        # Iterate over each systematic up/down pair
        for key in list(systs.keys()):
            if key.endswith("up"):
                base = key[:-2]  # strip 'up'
                down_key = base + "down"
                if down_key in systs:
                    up_counts = np.array(systs[key][hist_key]["counts"], dtype=float)
                    down_counts = np.array(systs[down_key][hist_key]["counts"], dtype=float)
                    theta = rng.normal(loc=0, scale=1) if rng else 0.0
                    shift = 0.5 * (up_counts - down_counts)
                    shape += theta * shift
        return edges, shape
    else:
        return edges, counts

def sum_background_histograms(pkl, hist_key, bkg_list, profile=False, rng=None):
    """Sum histograms from multiple background samples."""
    edges = None
    total_counts = None
    for sample in bkg_list:
        e, c = get_histogram(pkl, sample, hist_key, profile, rng)
        if edges is None:
            edges = e
            total_counts = np.zeros_like(c)
        elif not np.allclose(edges, e):
            LOGGER.error("Bin edges mismatch for sample '%s'", sample)
        total_counts += c
    return edges, total_counts

# ----------------------------------------------------------------------
# 3. ROI (Region-of-Interest) Utilities
# ----------------------------------------------------------------------
def cumulative(counts):
    """Compute cumulative sum of counts with a leading zero."""
    return np.concatenate(([0.0], np.cumsum(counts)))

def integral(cum, lo, hi):
    """Integral of counts between bin indices [lo, hi) using cumulative array."""
    return cum[hi] - cum[lo]

def enlarge_for_bkg(cnt_bkg, edges, lo, hi):
    """Expand ROI to ensure at least one background event inside."""
    bkg_cum = cumulative(cnt_bkg)
    if integral(bkg_cum, lo, hi) > 1.0:
        return lo, hi
    left, right = lo, hi
    n_bins = len(cnt_bkg)
    while True:
        can_left = left > 0 and edges[left-1] >= MIN_MASS_EDGE
        can_right = right < n_bins
        gain_left = cnt_bkg[left-1] if can_left else -np.inf
        gain_right = cnt_bkg[right] if can_right else -np.inf
        if gain_left >= gain_right and can_left:
            left -= 1
        elif can_right:
            right += 1
        else:
            break
        if integral(cumulative(cnt_bkg), left, right) > 1.0:
            LOGGER.debug("Enlarged ROI to include background: [%d, %d) bins", left, right)
            return left, right
    sys.exit("[FATAL] Could not build an 80% ALP window with non-zero background.")

def tight_window(cnt_sig, edges, i_cen, frac=0.8):
    """
    Build a greedy 80%-containment window around center i_cen symmetrically.
    """
    lo = hi = i_cen
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
        if hi < len(cnt_sig) - 1:
            hi += 1
            s_in += cnt_sig[hi]
        # If neither side can grow any further we are stuck → break early
        if lo == 0 and hi == len(cnt_sig) - 1:
            break
        LOGGER.debug("[tight_window] expand → bins=[%d,%d)  S_in=%g / %g = %.2f%%", lo, hi + 1, s_in, s_total, 100 * s_in / s_total)
    return lo, hi+1  # hi is exclusive

def find_best_roi(cnt_sig, cnt_bkg, edges):
    """Scan all bins and return ROI (lo, hi) with maximum expected Z."""
    mass_centers = 0.5 * (edges[:-1] + edges[1:])
    sig_cum = cumulative(cnt_sig)
    bkg_cum = cumulative(cnt_bkg)
    best_z = -np.inf
    best_lo = best_hi = -1
    for i, m in enumerate(mass_centers):
        if m < MIN_MASS_EDGE:
            continue
        lo, hi = tight_window(cnt_sig, edges, i)
        lo, hi = enlarge_for_bkg(cnt_bkg, edges, lo, hi)
        s_exp = integral(sig_cum, lo, hi)
        b_exp = integral(bkg_cum, lo, hi)
        if b_exp <= 0:
            continue
        z_exp = s_exp / math.sqrt(b_exp)
        LOGGER.debug(
        "ROI centre=%.2f GeV, bins=[%d,%d) (S=%.2f, B=%.2f, √B=%.3f, Z=%.3f)",
        m, lo, hi, s_exp, b_exp,
        math.sqrt(b_exp) if b_exp else float("nan"),
        z_exp,
        )
        if z_exp > best_z:
            best_lo, best_hi, best_z = lo, hi, z_exp
    if best_lo < 0:
        sys.exit("[FATAL] No valid ROI found above 5 GeV")
    LOGGER.info("Best ROI bins=[%d,%d) → expected Z=%.3f", best_lo, best_hi, best_z)
    return best_lo, best_hi, best_z

def find_full_roi(cnt_sig, cnt_bkg, edges):
    """Return full histogram range as ROI."""
    return 0, len(cnt_sig), None

def compute_coarse_binning(signal_counts, bin_edges, min_signal=0.01, merge_above=40.0):
    """
    Compute a coarse binning by merging bins above 'merge_above' until each
    combined bin has at least 'min_signal' signal counts.
    """
    new_edges = [bin_edges[0]]
    running = 0.0
    for i in range(len(signal_counts)):
        high_edge = bin_edges[i+1]
        running += signal_counts[i]
        # Before the threshold: preserve fine binning
        # After threshold: combine until signal exceeds threshold
        if high_edge <= merge_above or running >= min_signal:
            new_edges.append(high_edge)
            running = 0.0  # reset accumulator for the next bin
    # Safety check: ensure the final edge of the original binning is included
    if new_edges[-1] != bin_edges[-1]:
        new_edges.append(bin_edges[-1])
    return np.array(new_edges)

# ----------------------------------------------------------------------
# 4. Toy-MC Utilities
# ----------------------------------------------------------------------

def compute_asimov_significance(signal, background):
    """Asimov significance Z = sqrt(2[(s+b) log(1+s/b) - s])."""
    if background <= 0 or signal <= 0:
        return 0.0
    return math.sqrt(2 * ((signal + background) * math.log(1 + signal / background) - signal))

def get_random_from_hist(bin_edges, bin_contents, rng, n_samples):
    """
    Generate random samples from a histogram distribution defined by bin_contents.
    """
    # Ensure NumPy arrays with a float dtype for probability calculations
    bin_edges = np.array(bin_edges)
    bin_contents = np.array(bin_contents, dtype=float)
    # If the histogram is empty, return an array of zeros
    total = bin_contents.sum()
    if total <= 0:
        return np.zeros(n_samples)
    # Normalise bin contents to obtain probabilities
    probs = bin_contents / total
    cdf = np.cumsum(probs)  # cumulative distribution function in [0, 1]
    # Step 1: pick random CDF values
    random_vals = rng.random(n_samples)
    # Step 2: locate the first CDF entry that exceeds each random number
    bins = np.searchsorted(cdf, random_vals, side="right")
    # Guard against edge-cases where numerical precision pushes an index
    # slightly outside the valid range:
    #   • Lowest valid bin index  = 0
    #   • Highest valid bin index = (number of bins) - 1
    #     = (len(bin_edges) - 1)  - 1  ->  len(bin_edges) - 2
    bins = np.clip(bins, 0, len(bin_edges)-2)
    # Step 3: sample uniformly *within* the selected bin
    lows = bin_edges[bins]
    highs = bin_edges[bins+1]
    return lows + (highs - lows) * rng.random(n_samples)

def run_toys_likelihood_based(rng, S_fix, B_exp, n_trials, sigma, mode,
                              cnt_bkg=None, edges=None, lo=None, hi=None):
    """
    Run toy experiments using shape-aware Poisson fluctuations.
    mode: "s_plus_b", "bkg_only", or "s_scaled_plus_b".
    Skips toys where background in ROI is too small.
    """
    values = []
    attempts = 0
    max_attempts = 10 * n_trials    # not every toy can give bkg above MIN_B limit, give more try.
    MIN_B = 1
    while len(values) < n_trials and attempts < max_attempts:
        # Fluctuate background shape
        bkg_shape = rng.poisson(cnt_bkg)
        bkg_samp = get_random_from_hist(edges, bkg_shape, rng, n_samples=bkg_shape.sum())
        bkg_hist, _ = np.histogram(bkg_samp, bins=edges)
        b_roi = bkg_hist[lo:hi].sum()
        if b_roi < MIN_B:
            attempts += 1
            continue    # skip
        # Signal fluctuation
        if mode in ("s_plus_b", "s_scaled_plus_b"):
            # Determine signal count in ROI
            S_int = int(S_fix) if isinstance(S_fix, (int, float)) else int(np.sum(S_fix[lo:hi]))
            S_toy = rng.poisson(S_int)
            sig_samp = rng.uniform(edges[lo], edges[hi], size=S_toy)
            sig_hist, _ = np.histogram(sig_samp, bins=edges)
            s_roi = sig_hist[lo:hi].sum()
            z = compute_asimov_significance(s_roi, b_roi)
        else:
            z = math.sqrt(b_roi)
        #     s_roi = 0
        # z = compute_asimov_significance(s_roi, b_roi)
        values.append(z)
        attempts += 1
    if len(values) < n_trials:
        LOGGER.warning("Only %d toys generated (requested %d) with B >= %d", len(values), n_trials, MIN_B)
    return np.array(values)

def compute_cls95_limit(B_obs, B_exp, S_template, edges, lo, hi, rng, ntrials=10000):
    """
    Estimate the **95 % CLs** upper limit (σ\_obs) on a signal strength
    using the frequentist CLs method with simple Poisson toys.
    """
    from scipy.stats import poisson
    s_vals = np.linspace(0, 10 * S_template.sum(), 100)
    # -- Toy‐MC loop: evaluate CL_s for each hypothesis ---------------------------
    for s in s_vals:
        toys_sb = rng.poisson(B_exp + s, size=ntrials)  # S+B hypothesis
        toys_b  = rng.poisson(B_exp, size=ntrials)  # B‐only hypothesis
        # Compute *p*-values:
        #   p_sb = P(N ≥ B_obs | S+B),  p_b = P(N ≥ B_obs | B)
        p_sb = np.mean(toys_sb >= B_obs)
        p_b = np.mean(toys_b >= B_obs)
        cls = p_sb / p_b if p_b > 0 else 1.0    # protect against divide-by-zero
        if cls < 0.05:
            return s    # earliest crossing point is the limit
    # If the scan never crosses 0.05, return the largest tested signal strength
    return s_vals[-1]   

def compute_s95_likelihood(s, b, cl=0.95):
    # Simplified CLs-style s95 calculator using Poisson stats
    from scipy.stats import poisson
    limit = 0
    while poisson.cdf(limit, b + s) < cl:
        limit += 1
    return limit

# ----------------------------------------------------------------------
# 5. Plotting Helpers
# ----------------------------------------------------------------------
def plot_with_band(x, y, y_low, y_high, ylabel, filename):
    """Plot a line with optional error band and save to file."""
    plt.figure()
    plt.plot(x, y, marker='o', color="black")
    if y_low is not None and y_high is not None:
        plt.fill_between(x, y_low, y_high, alpha=0.3)
    plt.xlabel("ALP Mass [GeV]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_distribution(vals, median, lo, hi, path, mode, sig_label, sigma_scale):
    """Plot a single distribution (S+B or B-only) with median and 68% band."""
    vals = np.array(vals)
    finite = vals[np.isfinite(vals)]
    med = median
    p16, p84 = lo, hi
    plt.figure(figsize=(7,5))
    color = "deepskyblue" if mode == "s_plus_b" else "gray"
    plt.hist(finite, bins=150, histtype="stepfilled", alpha=0.75,
             color=color, edgecolor="black")
    plt.axvline(med, linestyle="--", color=color, label=f"median = {med:.2f}")
    plt.axvline(p16, color="k", linestyle=":", linewidth=1)
    plt.axvline(p84, color="k", linestyle=":", linewidth=1, label="68% band")
    xlabel = r"$Z = \sqrt{-2\log\lambda}$"
    plt.title(f"{mode.replace('_',' ')} for {sig_label} ({sigma_scale}σ fluct.)")
    plt.ylabel("Toy experiments")
    plt.xlabel(f"{xlabel} (80 % ALP window; $m_{{low}}≥{MIN_MASS_EDGE}$ GeV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    LOGGER.info("Saving plot → %s", path)
    plt.close()

def plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo, path, sig_label, sigma_scale):
    """Overlay plot comparing S+B and B-only distributions."""
    med_sb, p16_sb, p84_sb = stats_sb
    med_bo, p16_bo, p84_bo = stats_bo
    v_sb = np.array(vals_sb)[np.isfinite(vals_sb)]
    v_bo = np.array(vals_bo)[np.isfinite(vals_bo)]
    bins = np.histogram_bin_edges(np.concatenate([v_sb, v_bo]), bins=100)
    plt.figure(figsize=(8,5))
    plt.hist(v_bo, bins=bins, alpha=0.6, color="gray", label="B only", edgecolor="black")
    plt.hist(v_sb, bins=bins, alpha=0.5, color="deepskyblue", label="S+B", edgecolor="black")
    plt.axvline(med_bo, linestyle="--", color="gray", label=f"Median B-only = {med_bo:.2f}")
    plt.axvline(med_sb, linestyle="--", color="deepskyblue", label=f"Median S+B = {med_sb:.2f}")
    plt.axvspan(p16_bo, p84_bo, color="gray", alpha=0.2)
    plt.axvspan(p16_sb, p84_sb, color="deepskyblue", alpha=0.2)
    plt.xlabel(r"$Z$")
    plt.ylabel("Toy experiments")
    plt.title(f"Toy Significance for {sig_label}")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def print_limit_summary(sig_key, res):
    """Print a summary of limits for one signal."""
    print("===== LIMIT SUMMARY =====")
    print(f"Signal: {sig_key}")
    eff = res.get("eff", None)
    if eff is not None:
        print(f"  • Efficiency:         {eff:.3f}")
    print(f"  • σ_nominal:          {res.get('sigma_nominal', float('nan')):.4f} nb")
    # Asimov Exclusion
    if "sigma_exclusion" in res:
        print(f"  • Asimov Exclusion:   Z = {res.get('z_asimov', float('nan')):.2f},  σ_95% = {res['sigma_exclusion']:.4f} nb")
    
    # Discovery Reach (Z = 5)
    if "mu_5sigma" in res:
        print(f"  • Discovery (Z=5):     σ = {res['sigma_5sigma']:.4f} nb")
    if "sigma_discovery_lo" in res:
        print(f"     ±1σ band: [{res['sigma_discovery_lo']:.4f}, {res['sigma_discovery_hi']:.4f}] nb")
    if "sigma_discovery_lo2" in res:
        print(f"     ±2σ band: [{res['sigma_discovery_lo2']:.4f}, {res['sigma_discovery_hi2']:.4f}] nb")
    
    # Expected Toy Limit (s95)
    if "s95" in res:
        print(f"  • Expected s95:        {res['s95']:.2f} [−1σ = {res.get('s95_lo', float('nan')):.2f}, +1σ = {res.get('s95_hi', float('nan')):.2f}]")
    
    # Observed Limit
    if "sigma_observed" in res:
        print(f"  • Observed Limit:      σ_obs_95% = {res['sigma_observed']:.4f} nb")
    print()

# ----------------------------------------------------------------------
# 6. Process a single signal (SR only)
# ----------------------------------------------------------------------
def process_signal(args, pkl, sig_key):
    """
    Process one ALP signal: select ROI, compute Asimov Z, run toy experiments,
    determine expected/observed limits and discovery thresholds.
    """
    global asimov_results, results, s95_results
    rng = np.random.default_rng(args.seed)

    # 6a. Load histograms for signal and background
    edges_sig, cnt_sig = get_histogram(pkl, sig_key, args.hist, profile=args.profile_systematics, rng=rng)
    edges_bkg, cnt_bkg = sum_background_histograms(pkl, args.hist, args.bkg, profile=args.profile_systematics, rng=rng)
    if not np.allclose(edges_sig, edges_bkg):
        LOGGER.warning("Signal and background histograms have different bin edges.")

    # 6b. Determine ROI (80% signal window or full range)
    if args.roi == "full":
        lo, hi, z_exp = find_full_roi(cnt_sig, cnt_bkg, edges_sig)
    else:
        lo, hi, z_exp = find_best_roi(cnt_sig, cnt_bkg, edges_sig)
    LOGGER.info("[%s] ROI: [%.2f, %.2f] GeV (expected Z=%.3f)",
                sig_key, edges_sig[lo], edges_sig[hi], z_exp)

    # 6c. Optionally apply coarse binning
    if args.coarse_binning:
        # keep track of the current ROI edges *before* changing the binning
        roi_left_edge  = edges_sig[lo]
        roi_right_edge = edges_sig[hi]           # `hi` is exclusive

        coarse_edges = compute_coarse_binning(cnt_sig, edges_sig,
                                                min_signal=0.001, merge_above=40.0)
        LOGGER.debug("[%s] Using coarse binning edges: %s", sig_key, coarse_edges)

        bin_centers = 0.5 * (edges_sig[:-1] + edges_sig[1:])
        cnt_sig, _ = np.histogram(bin_centers, bins=coarse_edges, weights=cnt_sig)
        cnt_bkg, _ = np.histogram(bin_centers, bins=coarse_edges, weights=cnt_bkg)

        edges_sig = coarse_edges
        # remap ROI indices to the new (coarse) binning
        lo = np.searchsorted(edges_sig, roi_left_edge,  side="left")
        hi = np.searchsorted(edges_sig, roi_right_edge, side="right")

    # 6d. Compute signal and background counts in ROI
    S_fix = cnt_sig[lo:hi].sum()
    B_exp = cnt_bkg[lo:hi].sum()
    if S_fix == 0 or B_exp == 0:
        LOGGER.warning("[%s] Empty S or B in ROI, skipping.", sig_key)
        return
    LOGGER.info("[%s] S = %.3f, B = %.3f in ROI", sig_key, S_fix, B_exp)

    # 6e. Compute Asimov significance
    eff = pkl[sig_key].get("eff_sr", None)
    if eff is None:
        LOGGER.warning("[%s] Efficiency 'eff_sr' not found in pickle, using default 0.35", sig_key)
        eff = 0.35
    scaled_S = S_fix * args.sigscale
    z_asimov = compute_asimov_significance(scaled_S, B_exp)
    results[sig_key] = {}
    results[sig_key]["eff"] = eff
    results[sig_key]["z_asimov"] = z_asimov
    asimov_results[sig_key] = (scaled_S, B_exp, z_asimov)
    LOGGER.info("[%s] Asimov Z = %.2f", sig_key, z_asimov)

    # 6f. Expected s95 (toy with S+B fluctuations)
    vals = run_toys_likelihood_based(rng, scaled_S, B_exp, args.ntrials, args.sigma, "s_plus_b",
                                     cnt_bkg=cnt_bkg, edges=edges_sig, lo=lo, hi=hi)
    s95 = np.percentile(vals, 95)
    s68lo = np.percentile(vals, 16)
    s68hi = np.percentile(vals, 84)
    results[sig_key]["s95"] = s95
    results[sig_key]["s95_lo"] = s68lo
    results[sig_key]["s95_hi"] = s68hi
    results[sig_key]["sigma95"] = (s95 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    LOGGER.debug("[%s] Expected s95 = %.2f", sig_key, s95)

    # 6g. Expected exclusion (toy with B-only fluctuations)
    trials = []
    rejected = 0
    MIN_B = 1
    from scipy.stats import poisson
    # Keep generating toys until we have the requested statistics
    while len(trials) < args.ntrials:
        # 1) Poisson-fluctuate each background-template bin
        bkg_shape = rng.poisson(cnt_bkg)
        # 2) Draw pseudo-events from that fluctuated shape
        samples = get_random_from_hist(edges_sig, bkg_shape, rng, n_samples=bkg_shape.sum())
        # 3) Re-histogram the pseudo-events in the nominal binning
        toy_hist, _ = np.histogram(samples, bins=edges_sig)
        # 4) Background yield inside the ROI
        b_toy = toy_hist[lo:hi].sum()
        if b_toy < MIN_B:   # skip toys with too little background
            rejected += 1
            continue    # skip this toy
        # 5) Convert that background fluctuation into an s95 limit
        if args.use_cls:    # CLs (frequentist) method
            s95_trial = compute_cls95_limit(b_toy, B_exp, cnt_sig, edges_sig, lo, hi, rng, ntrials=100)
        else:   # one-sided likelihood method
            s95_trial = compute_s95_likelihood(0, b_toy)
        # 6) Store the toy result
        trials.append(s95_trial)
    if rejected > 0:
        LOGGER.info("[%s] Skipped %d trials with B < %d", sig_key, rejected, MIN_B)
    trials = np.array(trials)
    med, lo16, hi84, lo2, hi98 = np.percentile(trials, [50, 16, 84, 2.5, 97.5])
    s95_results[sig_key] = (med, lo16, hi84, lo2, hi98)
    sigma_excl = (med / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_lo1 = (lo16 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_hi1 = (hi84 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_lo2 = (lo2 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_hi2 = (hi98 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    results[sig_key]["sigma_exclusion"] = sigma_excl
    results[sig_key]["sigma_exclusion_lo"] = sigma_lo1
    results[sig_key]["sigma_exclusion_hi"] = sigma_hi1
    results[sig_key]["sigma_exclusion_lo2"] = sigma_lo2
    results[sig_key]["sigma_exclusion_hi2"] = sigma_hi2
    LOGGER.info("[%s] σ_95%% exclusion = %.3f nb (+1σ: %.3f, -1σ: %.3f)", sig_key, sigma_excl, sigma_hi1, sigma_lo1)

    # 6h. Nominal σ and μ95
    try:
        mass_val = int(sig_key.split("_")[1].replace("GeV", ""))
        sigma_nominal = alp_sigma_nb.get(mass_val)
    except Exception:
        sigma_nominal = None
    if sigma_nominal:
        results[sig_key]["sigma_nominal"] = sigma_nominal
        mu_95 = sigma_excl / sigma_nominal
        results[sig_key]["mu_95"] = mu_95
        LOGGER.info("[%s] μ_95 = %.2f", sig_key, mu_95)
    else:
        LOGGER.warning("[%s] Could not determine nominal σ", sig_key)

    # 6i. Observed limit (one toy experiment for observed B)
    B_obs = rng.poisson(B_exp)
    if B_obs < MIN_B:
        LOGGER.warning(f"[{sig_key}] B_obs too small ({B_obs:.2f}), using fallback {MIN_B}")
        B_obs = MIN_B
    if args.use_cls:
        LOGGER.info(f"[{sig_key}] Using CLs method for observed limit")
        s95_obs = compute_cls95_limit(B_obs, B_exp, cnt_sig, edges_sig, lo, hi, rng, ntrials=100)
    else:
        s95_obs = compute_s95_likelihood(0, B_obs)
    sigma_obs = (s95_obs / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    results[sig_key]["sigma_observed"] = sigma_obs
    LOGGER.info("[%s] Observed σ_95%% = %.3f nb (B_obs = %.0f)", sig_key, sigma_obs, B_obs)

    # 6j. Discovery threshold (Z_obs = 5) for observed B
    def find_mu_for_Z(B, target_Z=5.0, tol=1e-3):
        """Find μ such that Z_obs = target_Z for given B_obs."""
        mu = 1.0
        for _ in range(50):
            Z = compute_asimov_significance(mu * scaled_S, B)
            if abs(Z - target_Z) < tol:
                return mu
            mu *= target_Z / (Z if Z > 0 else 1.0)
        return mu
    mu_5sigma = find_mu_for_Z(B_obs, 5.0)
    sigma_5sigma = (mu_5sigma / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    results[sig_key]["mu_5sigma"] = mu_5sigma
    results[sig_key]["sigma_5sigma"] = sigma_5sigma
    LOGGER.info("[%s] Discovery threshold (Z=5): μ = %.2f, σ = %.3f nb", sig_key, mu_5sigma, sigma_5sigma)

    # 6k. Discovery threshold distribution (Z=2) bands
    mu_trials = []
    for _ in range(args.ntrials):
        b_toy = rng.poisson(B_exp)
        if b_toy > 0:
            mu_trial = 5 * math.sqrt(b_toy) / scaled_S
            mu_trials.append(mu_trial)
    mu_trials = np.array(mu_trials)
    if len(mu_trials):
        mu_med, mu_lo1, mu_hi1, mu_lo2, mu_hi2 = np.percentile(mu_trials, [50, 16, 84, 2.5, 97.5])
    else:
        mu_med = mu_lo1 = mu_hi1 = mu_lo2 = mu_hi2 = float("nan")
    results[sig_key]["mu_discovery_median"] = mu_med
    results[sig_key]["mu_discovery_lo1"] = mu_lo1
    results[sig_key]["mu_discovery_hi1"] = mu_hi1
    results[sig_key]["mu_discovery_lo2"] = mu_lo2
    results[sig_key]["mu_discovery_hi2"] = mu_hi2
    sigma_disc = (mu_med / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_disc_lo = (mu_lo1 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_disc_hi = (mu_hi1 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_disc_lo2 = (mu_lo2 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    sigma_disc_hi2 = (mu_hi2 / (eff * LUMINOSITY)) if eff > 0 else float("nan")
    results[sig_key]["sigma_discovery"] = sigma_disc
    results[sig_key]["sigma_discovery_lo"] = sigma_disc_lo
    results[sig_key]["sigma_discovery_hi"] = sigma_disc_hi
    results[sig_key]["sigma_discovery_lo2"] = sigma_disc_lo2
    results[sig_key]["sigma_discovery_hi2"] = sigma_disc_hi2
    LOGGER.info("[%s] σ_disc (Z=2) = %.3f nb", sig_key, sigma_disc)

    # 6l. Toy distribution plots
    if args.case == "overlay":
        vals_sb = run_toys_likelihood_based(rng, scaled_S, B_exp, args.ntrials, args.sigma,
                                           "s_plus_b", cnt_bkg=cnt_bkg, edges=edges_sig, lo=lo, hi=hi)
        vals_bo = run_toys_likelihood_based(rng, scaled_S, B_exp, args.ntrials, args.sigma,
                                           "bkg_only", cnt_bkg=cnt_bkg, edges=edges_sig, lo=lo, hi=hi)
        stats_sb = (np.median(vals_sb), *np.percentile(vals_sb, [16, 84]))
        stats_bo = (np.median(vals_bo), *np.percentile(vals_bo, [16, 84]))
        out_dir = Path("Significance_dis") / "sr" / sig_key
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_overlay(vals_sb, vals_bo, stats_sb, stats_bo,
                     out_dir / "overlay_distribution.png", sig_key, args.sigma)
        LOGGER.info("[%s] Saved overlay plot", sig_key)

    mode = args.case
    vals = run_toys_likelihood_based(rng, scaled_S, B_exp, args.ntrials, args.sigma,
                                     "s_plus_b" if mode=="s_plus_b" else "bkg_only",
                                     cnt_bkg=cnt_bkg, edges=edges_sig, lo=lo, hi=hi)
    med, p16, p84 = np.median(vals), *np.percentile(vals, [16, 84])
    out_dir = Path("Significance_dis") / "sr" / sig_key
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "s_plus_b" if mode == "s_plus_b" else "bkg_only"
    plot_distribution(vals, med, p16, p84, out_dir / f"{tag}_distribution.png",
                      mode, sig_key, args.sigma)
    LOGGER.info("[%s] Saved distribution plot", sig_key)

# ----------------------------------------------------------------------
# 7. Main Loop (SR only)
# ----------------------------------------------------------------------
def main():
    args = parse_cli()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.verbose:
        for noisy in ("matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.INFO)
    LOGGER.debug("CLI args: %s", args)

    if not PKL_PATH.is_file():
        sys.exit(f"[FATAL] Pickle not found: {PKL_PATH}")

    pkl_data = load_pickle(PKL_PATH)

    # Determine signals to process
    if args.signal.lower() in ("all", "alp_all"):
        sig_keys = sorted([k for k in pkl_data.keys() if k.startswith("alp_")])
    else:
        sig_keys = [args.signal]
    if not sig_keys:
        sys.exit("[FATAL] No ALP samples found in pickle.")
    LOGGER.info("Signals to process: %s", ", ".join(sig_keys))

    # Process each signal
    for sig in sig_keys:
        process_signal(args, pkl_data, sig)

    # Plot summary of Asimov Z vs mass (expected sensitivity)
    if asimov_results:
        masses, zvals = zip(*sorted((float(k.split("_")[1].replace("GeV","")), z) 
                                    for k, (_, _, z) in asimov_results.items()))
        plt.figure()
        plt.plot(masses, zvals, 'o-', color='blue')
        plt.xlabel("ALP Mass [GeV]")
        plt.ylabel("Expected Asimov Z")
        plt.title("Expected Sensitivity vs ALP Mass")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("expected_asimov_z_vs_mass.png")
        plt.close()
        LOGGER.info("Saved expected_asimov_z_vs_mass.png")

    # Plot expected s95 vs mass (toy MC)
    if s95_results:
        masses, s95_med, s95_lo, s95_hi = [], [], [], []
        for k, (med, lo, hi, lo2, hi2) in s95_results.items():
            try:
                m = float(k.split("_")[1].replace("GeV",""))
            except:
                continue
            masses.append(m)
            s95_med.append(med)
            s95_lo.append(med - lo)
            s95_hi.append(hi - med)
        if masses:
            order = np.argsort(masses)
            masses = np.array(masses)[order]
            s95_med = np.array(s95_med)[order]
            s95_lo = np.array(s95_lo)[order]
            s95_hi = np.array(s95_hi)[order]
            plt.figure()
            plt.errorbar(masses, s95_med, yerr=[s95_lo, s95_hi], fmt='o-', capsize=4)
            plt.xlabel("ALP Mass [GeV]")
            plt.ylabel("Expected $s_{95}$ (events)")
            plt.title("Expected $s_{95}$ vs ALP Mass")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("expected_s95_vs_mass.png")
            plt.close()
            LOGGER.info("Saved expected_s95_vs_mass.png")

    # Plot cross-section exclusion with ±1σ/±2σ bands
    if results:
        sig_sorted = sorted(results.keys(), key=lambda k: float(k.split("_")[1].replace("GeV","")))
        masses = [float(k.split("_")[1].replace("GeV","")) for k in sig_sorted]
        sigma_excl = np.array([results[k]["sigma_exclusion"] for k in sig_sorted])
        sigma_obs  = np.array([results[k]["sigma_observed"] for k in sig_sorted])
        sigma_lo1  = np.array([results[k]["sigma_exclusion_lo"] for k in sig_sorted])
        sigma_hi1  = np.array([results[k]["sigma_exclusion_hi"] for k in sig_sorted])
        sigma_lo2  = np.array([results[k]["sigma_exclusion_lo2"] for k in sig_sorted])
        sigma_hi2  = np.array([results[k]["sigma_exclusion_hi2"] for k in sig_sorted])
        yerr_lo_1 = sigma_excl - sigma_lo1
        yerr_hi_1 = sigma_hi1 - sigma_excl
        yerr_lo_2 = sigma_excl - sigma_lo2
        yerr_hi_2 = sigma_hi2 - sigma_excl
        band_lo1 = gaussian_filter1d(sigma_excl - yerr_lo_1, sigma=0.5)
        band_hi1 = gaussian_filter1d(sigma_excl + yerr_hi_1, sigma=0.5)
        band_lo2 = gaussian_filter1d(sigma_excl - yerr_lo_2, sigma=0.5)
        band_hi2 = gaussian_filter1d(sigma_excl + yerr_hi_2, sigma=0.5)
        plt.figure(figsize=(7,5))
        plt.plot(masses, sigma_excl, '-', color='black', label=r"Median σ$_{95}$ expected")
        plt.fill_between(masses, band_lo2, band_hi2, color='yellow', alpha=0.3, label=r"±2σ band")
        plt.fill_between(masses, band_lo1, band_hi1, color='green', alpha=0.4, label=r"±1σ band")
        plt.plot(masses, sigma_obs, 'k--', label=r"Observed σ$_{95}$")
        plt.xlabel("m$_{a}$ [GeV]")
        plt.ylabel("σ required for exclusion [nb]")
        plt.yscale("log")
        plt.title("Required σ for 95% CL exclusion vs ALP mass")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("required_sigma_exclusion_with_bands.png")
        plt.close()
        LOGGER.info("Saved required_sigma_exclusion_with_bands.png")

    # Final: print summary of limits
    print("\n===== ALL LIMIT SUMMARIES =====")
    for sig in sorted(results):
        print_limit_summary(sig, results[sig])

if __name__ == "__main__":
    main()
