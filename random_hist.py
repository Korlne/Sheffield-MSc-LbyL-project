#!/usr/bin/env python3
"""
random_hist.py
--------------------
Toy studies inside the 80 % ALP mass window **with Gaussian-fluctuated background**.

The *overlay* mode draws **both** the *s_plus_b*
(ALP + background) *and* *bkg_only* (background-only) significance distributions
on the same canvas **and** introduces two batch-processing shortcuts:

* ``--signal alp_all`` - iterate over **all** ALP-mass samples present in the
  histogram pickle,
* ``--aco cr&sr``      - process **both** acoplanarity regions in one go.

For every (aco, signal) pair the script reproduces the previous behaviour:
plots and summaries appear under

    Significance_dis/<aco>/<signal>/<tag>_80pc_distribution.png

The command-line below now sweeps *all* ALP masses in *both* regions and draws
overlay plots for each case::

    python3 random_hist.py --signal alp_all --aco cr&sr --case overlay
"""

# ----------------------------------------------------------------------
# 0. Imports, constants
# ----------------------------------------------------------------------
import argparse
import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

MIN_MASS_EDGE = 5.0  # ROI must start at ≥ 5 GeV

# ----------------------------------------------------------------------
# 1. CLI parsing
# ----------------------------------------------------------------------

def parse_cli():
    """Return the parsed command-line options."""
    p = argparse.ArgumentParser(
        description=(
            "Toy study inside the 80 % ALP window (Gaussian-fluctuated background)."
            "Supports batch processing via the special values:"
            "   --signal alp_all  → loop over every ALP mass in the pickle,"
            "   --aco    cr&sr    → run both acoplanarity regions."
        )
    )
    p.add_argument(
        "--aco",
        default="cr",
        help="Acoplanarity region: 'cr', 'sr' or 'cr&sr'/'both' to process both",
    )
    p.add_argument(
        "--signal",
        default="alp",
        help="Key of the ALP sample or 'alp_all' to process all masses found",
    )
    p.add_argument(
        "--bkg",
        nargs="+",
        default=["yyee", "cep", "signal"],
        help="Background sample keys (default: yyee cep signal)",
    )
    p.add_argument("--ntrials", type=int, default=10000, help="Number of toy experiments") # 10k trials for the standard statistical requirement
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument(
        "--sigma", type=float, default=1.0, help="Gaussian width scale (1 → σ = √B)"
    )
    p.add_argument("--plot", default=None, help="Output file name (auto-tagged if omitted)")
    p.add_argument(
        "--case",
        choices=["overlay", "s_plus_b", "bkg_only"],
        default="overlay",
        help=(
            "Study type: 'overlay' (default, plot both distributions), "
            "'s_plus_b' or 'bkg_only'",
        ),
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# 2. Helpers for pickle access
# ----------------------------------------------------------------------

def load_pickle(path):
    """Load the histogram pickle and strip auxiliary keys."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        sys.exit(f"[FATAL] Could not read pickle '{path}': {e}")
    data.pop("__scale_factors__", None)  # weights already scaled
    return data


def clean_events(vals, wts):
    """Drop events with non-finite value or weight and return np.ndarrays."""
    vals = np.asarray(vals, dtype=float)
    wts = np.asarray(wts, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(wts)
    return vals[mask], wts[mask]


def fetch_event_arrays(hist_pickle, sample, variable="mass"):
    """Return (edges, values, weights) for *sample/variable*."""
    try:
        h = hist_pickle[sample][variable]
    except KeyError:
        sys.exit(f"[FATAL] Missing '{sample}/{variable}' in pickle")
    edges = np.asarray(h["edges"])
    vals, wts = clean_events(h["values"], h["weights"])
    return edges, vals, wts


# ----------------------------------------------------------------------
# 3. ROI determination
# ----------------------------------------------------------------------

def cumulative(cnt):
    """Return zero-prepended cumulative sum."""
    return np.concatenate(([0.0], np.cumsum(cnt)))


def integral(cum, lo, hi):
    """Integral of *cnt* between inclusive *lo* and exclusive *hi* bin."""
    return cum[hi] - cum[lo]


def tight_window(cnt_sig, edges, i_cen, frac=0.80):
    """
    Define an initial window containing `frac` of the total signal (S).

    The window starts from a central bin `i_cen` and is grown symmetrically-ish
    by adding bins from the side with the higher signal content. This forms
    the first-pass 80 % signal window.
    """
    # --- initialisation -----------------------------------------------
    lo = hi = i_cen
    s_in = cnt_sig[i_cen]
    s_total = cnt_sig.sum()
    if s_total == 0:
        sys.exit("[FATAL] Signal histogram is empty - cannot build ROI")
    target = frac * s_total  # 80 % of total signal events

    # --- expand until ≥ 80 % of S is inside ---------------------------
    while s_in < target:
        can_left = lo > 0 and edges[lo - 1] >= MIN_MASS_EDGE
        gain_left = cnt_sig[lo - 1] if can_left else -np.inf
        gain_right = cnt_sig[hi] if hi < len(cnt_sig) - 1 else -np.inf

        # Prefer the side with more signal; break ties by favouring left.
        if gain_left >= gain_right and can_left:
            lo -= 1
            s_in += gain_left
        elif hi < len(cnt_sig) - 1:
            hi += 1
            s_in += gain_right
        else:
            # No further expansion possible (should be rare)
            break

    return lo, hi + 1  # hi is returned as *exclusive*


def enlarge_for_bkg(cnt_bkg, edges, lo, hi):
    """
    Expand a given window `(lo, hi)` to ensure it contains non-zero background (B).

    If the window already contains background, it is returned unchanged.
    Otherwise, the function expands the window one bin at a time, choosing
    the side with higher background content, until the condition B > 0 is met.
    The `MIN_MASS_EDGE` is always respected.

    If the entire mass range is exhausted and still no background is found,
    the program terminates, as a meaningful significance calculation is impossible.
    """
    bkg_cum = cumulative(cnt_bkg)

    # --- quick exit if background is already present ------------------
    if integral(bkg_cum, lo, hi) > 0.0:
        return lo, hi

    # --- otherwise grow the window -----------------------------------
    left = lo
    right = hi
    n_bins = len(cnt_bkg)

    while True:
        can_left = left > 0 and edges[left - 1] >= MIN_MASS_EDGE
        can_right = right < n_bins

        gain_left = cnt_bkg[left - 1] if can_left else -np.inf
        gain_right = cnt_bkg[right] if can_right else -np.inf

        # Choose the side with the greater potential background gain.
        if gain_left >= gain_right and can_left:
            left -= 1
        elif can_right:
            right += 1
        else:
            # No further expansion possible – give up.
            break

        if integral(bkg_cum, left, right) > 0.0:
            return left, right

    # If we reach this point, the entire allowed mass range still has zero bkg.
    sys.exit("[FATAL] Could not build an 80 % ALP window with non-zero background.")


def find_best_roi(cnt_sig, cnt_bkg, edges):
    """
    Find the best Region of Interest (ROI) by scanning all possible mass windows.

    The best ROI is the one that maximises the asymptotic expected significance
    `Z = S/√B`.

    For each possible central bin, this function first defines a candidate window
    that satisfies two conditions as per the analysis strategy:
    1. The window must contain ~80 % of the total signal events. This is
       achieved by starting with a central bin and expanding outwards (`tight_window`).
    2. The window must contain a non-zero number of background events. If the
       initial 80 % signal window has no background, it is expanded further
       until B > 0 (`enlarge_for_bkg`).

    Both expansion steps respect the `MIN_MASS_EDGE` constraint. The function
    then returns the window that yielded the highest `Z`.
    """
    mass_centres = 0.5 * (edges[:-1] + edges[1:])
    sig_cum = cumulative(cnt_sig)
    bkg_cum = cumulative(cnt_bkg)

    best_lo = best_hi = -1
    best_z = -np.inf

    for i_cen, m_cen in enumerate(mass_centres):
        if m_cen < MIN_MASS_EDGE:
            # Honour the lower mass threshold.
            continue

        # -- build a candidate window satisfying the two main conditions --
        lo, hi = tight_window(cnt_sig, edges, i_cen)
        lo, hi = enlarge_for_bkg(cnt_bkg, edges, lo, hi)

        # -- compute S, B, Z -------------------------------------------
        s_exp = integral(sig_cum, lo, hi)
        b_exp = integral(bkg_cum, lo, hi)
        z_exp = np.inf if b_exp == 0 else s_exp / math.sqrt(b_exp)

        # -- keep the best candidate -----------------------------------
        if z_exp > best_z:
            best_lo, best_hi, best_z = lo, hi, z_exp

    if best_lo < 0:
        sys.exit("[FATAL] No valid ROI satisfying the 80 %/bkg>0 constraints above 5 GeV")

    return best_lo, best_hi, best_z


# ----------------------------------------------------------------------
# 4. Gaussian fluctuation utility
# ----------------------------------------------------------------------

def fluctuate_total_gauss(n_tot, rng, sigma_scale):
    """Return a non-negative integer drawn from N(n_tot, sigma_scale·√n_tot)."""
    std = math.sqrt(n_tot) * sigma_scale
    n_fluct = int(round(rng.normal(n_tot, std)))
    return max(n_fluct, 0)


# ----------------------------------------------------------------------
# 5. Toy loop
# ----------------------------------------------------------------------

def run_toys(
    rng,
    bkg_info,
    lo_edge,
    hi_edge,
    s_roi_fixed,
    ntrials,
    sigma_scale,
    mode,
):
    """Return an array of *ntrials* test statistics."""
    vals = np.empty(ntrials, float)
    bkg_n_tot = [len(v) for _, v, _ in bkg_info]

    for itoy in range(ntrials):
        # ---- background --------------------------------------------------
        B_roi = 0.0
        for (lbl, v_arr, w_arr), n_tot in zip(bkg_info, bkg_n_tot):
            n_draw = fluctuate_total_gauss(n_tot, rng, sigma_scale)
            if n_draw == 0:
                continue
            idx = rng.integers(0, n_tot, size=n_draw)  # bootstrap with replacement
            sel = (v_arr[idx] >= lo_edge) & (v_arr[idx] < hi_edge)
            B_roi += w_arr[idx][sel].sum()

        # ---- store test statistic ---------------------------------------
        if mode == "bkg_only":
            vals[itoy] = math.sqrt(B_roi)
        else:  # s_plus_b
            vals[itoy] = np.inf if B_roi == 0 else s_roi_fixed / math.sqrt(B_roi)

    return vals


# ----------------------------------------------------------------------
# 6. Plot helpers
# ----------------------------------------------------------------------

def plot_distribution(vals, med, p16, p84, plot_path, case, signal, sigma_scale):
    """Histogram of *vals* with non‑finite entries stripped off."""
    vals_plot = vals[np.isfinite(vals)]   # drop inf / nan

    plt.figure(figsize=(7, 5))
    color = "deepskyblue" if case == "s_plus_b" else "grey"
    plt.hist(vals_plot, bins="auto", histtype="stepfilled", alpha=0.75,
             color=color)
    plt.axvline(med, ls="--", lw=1.2, label=f"median = {med:.2f}")
    plt.axvline(p16, color="k", ls=":", lw=1)
    plt.axvline(p84, color="k", ls=":", lw=1, label="68 % band")

    xlabel = r"$Z = \dfrac{S}{\sqrt{B}}$" if case == "s_plus_b" else r"$\sqrt{B}$"
    title = (
        f"Significance distribution for ALP ma={signal} {sigma_scale}σ fluctuation"
        if case == "s_plus_b" else
        f"Significance distribution for ALP ma={signal} bkg only {sigma_scale}σ"
    )
    plt.title(title)
    plt.xlabel(f"{xlabel}  (80 % ALP window; $m_\text{{low}}≥{MIN_MASS_EDGE}$ GeV)")
    plt.ylabel("Toy experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()


def plot_overlay(
    vals_sb, vals_bo,
    med_sb, med_bo,
    p16_sb, p84_sb,
    p16_bo, p84_bo,
    plot_path, signal, sigma_scale,
):
    """Overlay histograms after removing inf/nan entries."""
    vals_sb_plot = vals_sb[np.isfinite(vals_sb)]  # filter
    vals_bo_plot = vals_bo[np.isfinite(vals_bo)]  # ilter

    plt.figure(figsize=(7, 5))

    # --- S + B ---------------------------------------------------------
    plt.hist(vals_sb_plot, bins="auto", histtype="stepfilled", alpha=0.6,
             color="deepskyblue", label="S plus B")
    plt.axvline(med_sb, ls="--", lw=1.2, color="deepskyblue",
                label=f"median s+b = {med_sb:.2f}")
    plt.axvline(p16_sb, color="deepskyblue", ls=":", lw=1)
    plt.axvline(p84_sb, color="deepskyblue", ls=":", lw=1,
                label="68 % band (s+b)")

    # --- B only --------------------------------------------------------
    plt.hist(vals_bo_plot, bins="auto", histtype="stepfilled", alpha=0.5,
             color="grey", label="bkg_only")
    plt.axvline(med_bo, ls="--", lw=1.2, color="grey",
                label=f"median bkg = {med_bo:.2f}")
    plt.axvline(p16_bo, color="grey", ls=":", lw=1)
    plt.axvline(p84_bo, color="grey", ls=":", lw=1,
                label="68 % band (bkg)")

    plt.title(f"S+B and bkg-only significance for ALP ma={signal}"
              f" ({sigma_scale}σ fluctuation)")
    plt.xlabel(r"Significance statistic  (80 % ALP window; $m_{\text{low}}≥"
               f"{MIN_MASS_EDGE}$ GeV)")
    plt.ylabel("Toy experiments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()



# ----------------------------------------------------------------------
# 7. Per-signal/region worker
# ----------------------------------------------------------------------

def process_signal_region(args, region, signal_key):
    """Run the full study for one (aco, signal) pair."""

    # Reset RNG for each signal to ensure consistent behavior
    rng = np.random.default_rng(args.seed)

    # ---  load pickle ---------------------------------------------------
    pickle_path = os.path.join(f"Plots_sr_{region}", f"histograms_{region}.pkl")
    hist_pickle = load_pickle(pickle_path)

    # ---  extract signal & background ----------------------------------
    edges_ref, val_sig, wt_sig = fetch_event_arrays(hist_pickle, signal_key)
    bkg_info = []
    for lbl in args.bkg:
        e_tmp, v_tmp, w_tmp = fetch_event_arrays(hist_pickle, lbl)
        if not np.array_equal(e_tmp, edges_ref):
            sys.exit(f"[FATAL] Binning of '{lbl}' differs from signal binning")
        bkg_info.append((lbl, v_tmp, w_tmp))

    # ---  enhanced debugging for signal weights ------------------------
    raw_events = len(hist_pickle[signal_key]["mass"]["values"])
    cleaned_events = len(val_sig)
    
    print(f"[DEBUG] {region}/{signal_key}: Raw events: {raw_events}, After cleaning: {cleaned_events}")
    if cleaned_events > 0:
        print(f"[DEBUG] Weight stats: min={wt_sig.min():.6e}, max={wt_sig.max():.6e}, mean={wt_sig.mean():.6e}")
    
    # ---  build weighted histograms ------------------------------------
    cnt_sig_w, _ = np.histogram(val_sig, bins=edges_ref, weights=wt_sig)
    cnt_bkg_w = np.zeros_like(cnt_sig_w, dtype=float)
    for _, v, w in bkg_info:
        h_tmp, _ = np.histogram(v, bins=edges_ref, weights=w)
        cnt_bkg_w += h_tmp

    total_sig_weight = cnt_sig_w.sum()
    print(f"[DEBUG] {region}/{signal_key}: Total weighted sum: {total_sig_weight:.6e}")

    # Check if signal histogram is effectively empty
    if total_sig_weight <= 1e-10:
        print(f"[WARNING] Skipping {region}/{signal_key}: Signal histogram is effectively empty\n")
        return

    # ---  find best ROI -------------------------------------------------
    lo_idx, hi_idx, z_exp = find_best_roi(cnt_sig_w, cnt_bkg_w, edges_ref)
    lo_edge, hi_edge = edges_ref[lo_idx], edges_ref[hi_idx]
    print(
        f"[{region}/{signal_key}] 80 % ALP window: [{lo_edge:.2f}, {hi_edge:.2f}] GeV (expected Z = {z_exp:.3f})"
    )

    # ---  fixed signal yield -------------------------------------------
    sel_sig_roi = (val_sig >= lo_edge) & (val_sig < hi_edge)
    S_roi_fixed = wt_sig[sel_sig_roi].sum()

    # ------------------------------------------------------------------
    #  overlay  
    # ------------------------------------------------------------------
    if args.case == "overlay":
        # -- run toys ----------------------------------------------------
        vals_sb = run_toys(
            rng,
            bkg_info,
            lo_edge,
            hi_edge,
            S_roi_fixed,
            args.ntrials,
            args.sigma,
            "s_plus_b",
        )
        vals_bo = run_toys(
            rng,
            bkg_info,
            lo_edge,
            hi_edge,
            S_roi_fixed,
            args.ntrials,
            args.sigma,
            "bkg_only",
        )

        # -- summaries ---------------------------------------------------
        vals_sb = vals_sb[np.isfinite(vals_sb)]
        vals_bo = vals_bo[np.isfinite(vals_bo)]
    
        med_sb  = np.median(vals_sb)
        med_bo  = np.median(vals_bo)
        p16_sb,  p84_sb  = np.percentile(vals_sb, [16, 84])
        p16_bo,  p84_bo  = np.percentile(vals_bo, [16, 84])
        p025_sb, p975_sb = np.percentile(vals_sb, [2.5, 97.5])
        p025_bo, p975_bo = np.percentile(vals_bo, [2.5, 97.5])

        print("[s_plus_b]  Trials: {:d}  Median Z: {:.3f}  68 %: [{:.3f}, {:.3f}]  95 %: [{:.3f}, {:.3f}]".format(
            args.ntrials, med_sb, p16_sb, p84_sb, p025_sb, p975_sb
        ))
        print("[bkg_only]  Trials: {:d}  Median √B: {:.3f}  68 %: [{:.3f}, {:.3f}]  95 %: [{:.3f}, {:.3f}]".format(
            args.ntrials, med_bo, p16_bo, p84_bo, p025_bo, p975_bo
        ))

        # -- plot --------------------------------------------------------
        out_dir = os.path.join("Significance_dis", region, signal_key)
        os.makedirs(out_dir, exist_ok=True)
        tag = "overlay"
        fname = os.path.basename(args.plot) if args.plot else f"{tag}_80pc_distribution.png"
        plot_path = os.path.join(out_dir, fname)
        plot_overlay(
            vals_sb,
            vals_bo,
            med_sb,
            med_bo,
            p16_sb,
            p84_sb,
            p16_bo,
            p84_bo,
            plot_path,
            signal_key,
            args.sigma,
        )
        print(f"[OK] Plot saved as '{plot_path}'\n")
        return

    # ------------------------------------------------------------------
    #  single-case handling 
    # ------------------------------------------------------------------
    vals = run_toys(
        rng,
        bkg_info,
        lo_edge,
        hi_edge,
        S_roi_fixed,
        args.ntrials,
        args.sigma,
        args.case,
    )

    med = np.median(vals)
    p16, p84 = np.percentile(vals, [16, 84])
    p025, p975 = np.percentile(vals, [2.5, 97.5])

    if args.case == "s_plus_b":
        print(
            "  Trials: {:d}  Median Z: {:.3f}  68 %: [{:.3f}, {:.3f}]  95 %: [{:.3f}, {:.3f}]".format(
                args.ntrials, med, p16, p84, p025, p975
            )
        )
    else:
        print(
            "  Trials: {:d}  Median √B: {:.3f}  68 %: [{:.3f}, {:.3f}]  95 %: [{:.3f}, {:.3f}]".format(
                args.ntrials, med, p16, p84, p025, p975
            )
        )

    # ---  plot ---------------------------------------------------------
    out_dir = os.path.join("Significance_dis", region, signal_key)
    os.makedirs(out_dir, exist_ok=True)
    tag = "alp" if args.case == "s_plus_b" else "bkg_only"
    fname = os.path.basename(args.plot) if args.plot else f"{tag}_80pc_distribution.png"
    plot_path = os.path.join(out_dir, fname)
    plot_distribution(vals, med, p16, p84, plot_path, args.case, signal_key, args.sigma)
    print(f"[OK] Plot saved as '{plot_path}'\n")


# ----------------------------------------------------------------------
# 8. Main routine (loops over requested regions & signals)
# ----------------------------------------------------------------------

def main():
    args = parse_cli()

    # Determine list of acoplanarity regions --------------------------------
    if args.aco.lower() in {"cr&sr", "both", "all"}:
        aco_regions = ["cr", "sr"]
    else:
        aco_regions = [args.aco.lower()]

    for region in aco_regions:
        # Discover signal samples if 'alp_all' selected --------------------
        if args.signal.lower() in {"alp_all", "all"}:
            pickle_path = os.path.join(f"Plots_sr_{region}", f"histograms_{region}.pkl")
            hist_pickle = load_pickle(pickle_path)
            signals = sorted([k for k in hist_pickle.keys() if k.startswith("alp")])
            if not signals:
                sys.exit(f"[FATAL] No keys starting with 'alp' found in pickle '{pickle_path}'")
        else:
            signals = [args.signal]

        for sig_key in signals:
            process_signal_region(args, region, sig_key)


if __name__ == "__main__":
    main()