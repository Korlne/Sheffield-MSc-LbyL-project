#!/usr/bin/env python3
# =============================================================================
#  Plot-driver for the Light-by-Light (LbyL) analysis                (plots.py)
#  ============================================================================
#
#  What this file does
#  -------------------
#  • Reads **cut-flow pickles** written by `run_batch.py` *and* the “official”
#    binned-histogram pickles produced during reconstruction
#      – standard MC: *signal*, *cep*, γγ→e⁺e⁻ (*yyee*)  
#      – any number of ALP mass points (*alp4 … alp100*)  
#      – optional Run-2 overlay (data only; no weighting)
#  • Computes **sample-by-sample luminosity scale factors**  
#      – SM signal: analytical cross-section (879.2621 nb)  
#      – CEP: control-region normalisation against data  
#      – γγ→e⁺e⁻: fixed to the data CR yield  
#      – ALPs: σ(m<sub>a</sub>) lookup table → `alp_sigma_nb`
#  • Generates two plotting modes (CLI switch `--plot`):  
#      1. **stacked**  – four variable distributions  \
#         (m<sub>γγ</sub>, p<sub>T</sub><sup>γγ</sup>, A<sub>φ</sub>, E<sub>T</sub><sup>lead γ</sup>) + √N error hatch  
#      2. **cutflow**  – stepped, log-scaled event-flow comparison for up to
#         three MC benchmarks (signal, CEP, ALP5 by default)
#  • Writes all images into an auto-created directory  
#        `Plots_sr_<slice>`  or  `Plots_cr`
#  • Persists **fully-binned histograms** for *every* sample to
#        `histograms_<slice>.pkl`
#
#  Histogram-pickle layout
# dict_keys(['__scale_factors__', 'cep', 'signal', 'alp12', 'alp6', 'alp80', 'alp10', 'alp9', 'alp90', 'alp18', 'alp7', 'alp5', 'alp20', 'alp60', 'alp40', 'alp16', 'alp50', 'alp14', 'alp100', 'alp4', 'alp15', 'alp70', 'alp8', 'yyee', 'run2data'])
#  -----------------------
#      {
#        "__scale_factors__": {sample → float},
#        "<sample>": {
#            "mass":       {"counts": ndarray, "edges": ndarray,
#                           "values": ndarray | None, "weights": ndarray | None},
#            "pt":         {…},
#            "acop":       {…},
#            "leading_et": {…}
#        },
#        …
#      }
#  • *counts/edges* are always present.  
#  • *values/weights* are **only** filled for samples that come from
#    event-arrays (CEP, signal, ALPs).  Pre-binned inputs (yyee, Run-2) set
#    them to **None**.
#
#  Cut-flow-pickle layout (input)
#  ------------------------------
#  Identical to the structure returned by `lbyl_common.process_events`
#  (see that module for full schema).
#
#  Command-line quick-start
#  ------------------------
#  ```bash
#  # Signal-region slice  Aφ ≤ 0.010 
#  $ python3 plots.py --acomode sr --cut 0p010             # stacked (default)
#
#  # Same slice – cut-flow comparison only
#  $ python3 plots.py --acomode sr --cut 0p010 --plot cutflow
#
#  # Control region, skip Run-2 overlay
#  $ python3 plots.py --acomode cr --no-run2
#
#  # Produce every SR slice (0p005 … 0p020) in one go
#  $ python3 plots.py --acomode sr --cut all
#  ```
#
#  Dependencies
#  ------------
# ATLAS Athena AnalysisBase,25.2.15
# AlmaLinux 9
#
#  Extending / tweaking
#  --------------------
#  • To add another benchmark to the cut-flow plot, modify `wanted` /
#    `label_base` inside `plot_cutflows`.  
#  • To plot additional variables, extend `variable_bins`, `key_map`, and
#    the `plot_specs` list inside `plot_histograms`.
#
#  Last updated : 24 Jun 2025
# =============================================================================


import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from AthenaCommon.SystemOfUnits import nanobarn

# ----------------------------------------------------------------------
# Debug option (set from the --debug CLI switch)
# ----------------------------------------------------------------------
DEBUG = True          # ← default off

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# These **three** samples are stacked in the plots.
samples = ["yyee", "cep", "signal"]

# Lookup table for ALP σ (nb) as a simple dict (mass → σ)
alp_sigma_nb = {
    4: 7.967330e3,
    5: 6.953744e3,
    6: 6.044791e3,
    7: 5.300250e3,
    8: 4.670220e3,
    9: 4.154600e3,
    10: 3.709976e3,
    12: 3.016039e3,
    14: 2.499097e3,
    15: 2.285133e3,
    16: 2.093761e3,
    18: 1.782345e3,
    20: 1.526278e3,
    30: 7.77903e2,
    40: 4.36836e2,
    50: 2.600118e2,
    60: 1.604056e2,
    70: 1.016849e2,
    80: 6.546058e1,
    90: 4.280824e1,
    100: 2.824225e1,
}

# Use ATLASAlt style from mplhep.
hep.style.use(hep.style.ATLASAlt)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def load_cutflow_data(aco_mode):
    """Load cut-flow pickles for the standard samples **and** any ALPs."""
    data = {}

    # Standard samples – yyee is handled separately later
    for sample in ("cep", "signal"):
        fname = f"cutflow_{sample}_aco-{aco_mode}.pkl"
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                data[sample] = pickle.load(f)
        else:
            print(f"Cut-flow file {fname} not found. Skipping {sample}.")

    # Discover ALP samples (cutflow_alpX_aco-*.pkl)
    alp_paths = glob.glob(f"cutflow_alp*_aco-{aco_mode}.pkl")
    for path in alp_paths:
        sample_name = (
            os.path.basename(path)
            .replace("cutflow_", "")
            .replace(f"_aco-{aco_mode}.pkl", "")  # e.g. alp5
        )
        with open(path, "rb") as f:
            data[sample_name] = pickle.load(f)
        print(f"Loaded {sample_name} from {path}.")

    return data


def add_atlas_label(com=5.36):
    hep.atlas.label(
        label="",
        loc=4,
        rlabel=r"Pb+Pb $\sqrt{s_{NN}} = " + f"{com}" + r"\,\mathrm{TeV}$",
        year=2025,
    )


def add_error_band(hist_list, bins, hatch_pattern="xxxxx"):
    """Overlay a 1 σ (√N) error band for the stacked MC histograms."""
    stacked_counts = np.sum([h[0] for h in hist_list], axis=0)
    stacked_errors = np.sqrt(stacked_counts)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.fill_between(
        bin_centers,
        stacked_counts - stacked_errors,
        stacked_counts + stacked_errors,
        step="mid",
        facecolor="none",
        hatch=hatch_pattern,
        edgecolor="k",
        linewidth=0,
        alpha=0.5,
        label="Syst. uncertainty",
    )


# ----------------------------------------------------------------------
# Scale-factor calculation for CR
# ----------------------------------------------------------------------
def compute_CR_scale_factors(data):
    """Return a dict of luminosity scale factors for every sample."""
    dataLumi = 1.67 / nanobarn  # ≈ 1.67 nb⁻¹

    # --- Signal (SM light-by-light) -----------------------------------
    cross_section_signal = 879.2621 * nanobarn
    total_signal_events = data["signal"]["cut_flow"].get("Total Events", 1.0)
    exp_lbyl = (dataLumi * cross_section_signal) / total_signal_events
    N_sig_CR = 8.14056022664
    # N_sig_SR = 49.67461574918
    
    signal_scale = N_sig_CR/data["signal"]["cut_flow"]["Pass Acoplanarity Selection"]
    print("Computed global signal scale:", exp_lbyl)

    # --- CEP (γγ from gg) – control-region normalisation ---------------   
    # N_data_CR = 219
    # N_yyee_CR = 70
    
    # N_data_CR = 52.0
    
    N_cep_CR = 35.21717889374149
    # N_cep_SR = 9.321213574229608

    N_yyee_CR = 8.64226087961851
    # N_yyee_SR = 11.715064747927313
    
    
    cep_scale = N_cep_CR / data["cep"]["cut_flow"]["Pass Acoplanarity Selection"]
    print("Computed cep scaling factor:", cep_scale)
    
    # N_sig_CR = total_signal_events * signal_scale
    # N_cep_mc_CR = data["cep"]["cut_flow"]["Pass Acoplanarity Selection"]
    # N_cep_mc_CR = data["cep"]["cut_flow"].get("Total Events", 1.0)
    # print('number of N_cep_mc_CR = ', N_cep_mc_CR)
    # N_cep_data = N_data_CR - N_yyee_CR - (N_sig_CR * signal_scale)
    # cep_scale = N_cep_data / N_cep_mc_CR

    # --- yyee (γγ → e⁺e⁻) ---------------------------------------------
    N_yyee_mc_CR = np.sum(data["yyee"]["h_ZMassZoom"]["counts"])
    yyee_scale = N_yyee_CR / N_yyee_mc_CR if N_yyee_mc_CR > 0 else 1.0

    print("  → LbyL control-region scale factor:", signal_scale)
    print("  → CEP control-region scale factor:", cep_scale)
    print("  → yyee control-region scale factor:", yyee_scale)

    # ------------------------------------------------------------------
    # Assemble dictionary
    # ------------------------------------------------------------------
    scale_factors = {
        "yyee": yyee_scale,
        "cep": cep_scale,
        "signal": signal_scale,
    }

    # --- ALP samples ---------------------------------------------------
    for sample_name in data:
        if not sample_name.startswith("alp"):
            continue

        # Extract mass point, e.g. "alp5" → 5 GeV
        try:
            mass_point = int(sample_name.replace("alp", ""))
        except ValueError:
            print(f"Could not parse mass point from {sample_name}. Using unity scale.")
            scale_factors[sample_name] = 1.0
            continue

        sigma_nb = alp_sigma_nb.get(mass_point)
        if sigma_nb is None:
            print(f"No cross-section found for {sample_name}. Using unity scale.")
            scale_factors[sample_name] = 1.0
            continue

        total_events_alp = data[sample_name]["cut_flow"].get("Total Events", 1.0)
        scale_factors[sample_name] = dataLumi * (sigma_nb * nanobarn) / total_events_alp
        print(
            f"  → {sample_name}: σ={sigma_nb} nb, N_gen={total_events_alp}, "
            f"scale={scale_factors[sample_name]:.4e}"
        )

    return scale_factors

# ----------------------------------------------------------------------
# Scale-factor calculation (factored-out so both plotting modes can reuse it)
# ----------------------------------------------------------------------
def compute_SR_scale_factors(data):
    """Return a dict of luminosity scale factors for every sample."""
    dataLumi = 1.67 / nanobarn  # ≈ 1.67 nb⁻¹

    # --- Signal (SM light-by-light) -----------------------------------
    cross_section_signal = 879.2621 * nanobarn
    total_signal_events = data["signal"]["cut_flow"].get("Total Events", 1.0)
    exp_lbyl = (dataLumi * cross_section_signal) / total_signal_events
    # N_sig_CR = 8.14056022664
    N_sig_SR = 49.67461574918
    
    signal_scale = N_sig_SR/data["signal"]["cut_flow"]["Pass Acoplanarity Selection"]
    print("Computed global signal scale:", exp_lbyl)

    # --- CEP (γγ from gg) – control-region normalisation ---------------   
    # N_data_CR = 219
    # N_yyee_CR = 70
    
    N_data_CR = 52.0
    
    # N_cep_CR = 35.21717889374149
    N_cep_SR = 9.321213574229608

    # N_yyee_CR = 8.64226087961851
    N_yyee_SR = 11.715064747927313
    
    
    cep_scale = N_cep_SR / data["cep"]["cut_flow"]["Pass Acoplanarity Selection"]

    
    # N_sig_CR = total_signal_events * signal_scale
    # N_cep_mc_CR = data["cep"]["cut_flow"]["Pass Acoplanarity Selection"]
    # N_cep_mc_CR = data["cep"]["cut_flow"].get("Total Events", 1.0)
    # print('number of N_cep_mc_CR = ', N_cep_mc_CR)
    # N_cep_data = N_data_CR - N_yyee_CR - (N_sig_CR * signal_scale)
    # cep_scale = N_cep_data / N_cep_mc_CR

    # --- yyee (γγ → e⁺e⁻) ---------------------------------------------
    N_yyee_mc_SR = np.sum(data["yyee"]["h_ZMassZoom"]["counts"])
    yyee_scale = N_yyee_SR / N_yyee_mc_SR if N_yyee_mc_SR > 0 else 1.0

    print("  → LbyL control-region scale factor:", signal_scale)
    print("  → CEP control-region scale factor:", cep_scale)
    print("  → yyee control-region scale factor:", yyee_scale)

    # ------------------------------------------------------------------
    # Assemble dictionary
    # ------------------------------------------------------------------
    scale_factors = {
        "yyee": yyee_scale,
        "cep": cep_scale,
        "signal": signal_scale,
    }

    # --- ALP samples ---------------------------------------------------
    for sample_name in data:
        if not sample_name.startswith("alp"):
            continue

        # Extract mass point, e.g. "alp5" → 5 GeV
        try:
            mass_point = int(sample_name.replace("alp", ""))
        except ValueError:
            print(f"Could not parse mass point from {sample_name}. Using unity scale.")
            scale_factors[sample_name] = 1.0
            continue

        sigma_nb = alp_sigma_nb.get(mass_point)
        if sigma_nb is None:
            print(f"No cross-section found for {sample_name}. Using unity scale.")
            scale_factors[sample_name] = 1.0
            continue

        total_events_alp = data[sample_name]["cut_flow"].get("Total Events", 1.0)
        scale_factors[sample_name] = dataLumi * (sigma_nb * nanobarn) / total_events_alp
        print(
            f"  → {sample_name}: σ={sigma_nb} nb, N_gen={total_events_alp}, "
            f"scale={scale_factors[sample_name]:.4e}"
        )

    return scale_factors

# ----------------------------------------------------------------------
# Stacked-histogram plotting 
# ----------------------------------------------------------------------
def plot_histograms(data, plot_dir, aco_mode):
    """Stacked variable plots and histogram-pickle dump."""

    # ------------------------------------------------------------------
    # Bin definitions – default or taken from yyee pickle
    # ------------------------------------------------------------------
    if (
        "yyee" in data
        and isinstance(data["yyee"], dict)
        and "h_ZMassZoom" in data["yyee"]
    ):
        yd = data["yyee"]
        mass_bins = np.asarray(yd["h_ZMassZoom"]["edges"])
        pt_bins = np.asarray(yd["h_ZptZoom"]["edges"])
        acop_bins = np.asarray(yd["h_ZAcoZoom"]["edges"])
        leading_et_bins = np.asarray(yd["h_ZLeadingPhotonET"]["edges"])
    else:
        mass_bins = np.linspace(0, 30, 31)
        pt_bins = np.linspace(0, 5, 51)
        acop_bins = np.linspace(0, 0.1, 101)
        leading_et_bins = np.linspace(0, 15, 31)

    variable_bins = {
        "mass": mass_bins,
        "pt": pt_bins,
        "acop": acop_bins,
        "leading_et": leading_et_bins,
    }

            
    # ------------------------------------------------------------------
    # Global scale factors
    # ------------------------------------------------------------------
    scale_factors = compute_CR_scale_factors(data) if aco_mode == 'cr' else compute_SR_scale_factors(data)

    # ------------------------------------------------------------------
    # Histogram-building helpers
    # ------------------------------------------------------------------
    key_map = {
        "mass": "h_ZMassZoom",
        "pt": "h_ZptZoom",
        "acop": "h_ZAcoZoom",
        "leading_et": "h_ZLeadingPhotonET",
    }

    def build_hist(sample, variable, bins):
        """Return (counts, bins) for *any* sample."""
        scale = scale_factors.get(sample, 1.0)

        # Pre-binned samples (yyee & run-2 data)
        if key_map[variable] in data[sample]:
            obj = data[sample][key_map[variable]]
            return np.asarray(obj["counts"]) * scale, bins

        # Array-based samples (cep, signal, alpX …)
        res = data[sample]
        arrays = {
            "mass": res["diphoton_masses"],
            "pt": res["diphoton_pts"],
            "acop": res["diphoton_acoplanarity"],
            "leading_et": res["leading_photon_ets"],
        }
        values = np.asarray(arrays[variable])
        weights = np.asarray(res.get("event_weights", np.ones_like(values))) * scale
        counts, _ = np.histogram(values, bins=bins, weights=weights)
        return counts, bins

    def get_values_weights(sample, variable):
        """Return (values, weights) **after** scale factor application.

        For pre-binned samples the information is unavailable → (None, None).
        """
        scale = scale_factors.get(sample, 1.0)

        # Pre-binned samples – no event-level arrays available
        if key_map[variable] in data[sample]:
            return None, None

        res = data[sample]
        arrays = {
            "mass": res["diphoton_masses"],
            "pt": res["diphoton_pts"],
            "acop": res["diphoton_acoplanarity"],
            "leading_et": res["leading_photon_ets"],
        }
        v = np.asarray(arrays[variable])
        w = np.asarray(res.get("event_weights", np.ones_like(v))) * scale
        return v, w

    # ------------------------------------------------------------------
    # Build histograms for stacking (only yyee, cep, signal)
    # ------------------------------------------------------------------
    mass_hists = [build_hist(s, "mass", mass_bins) for s in samples]
    pt_hists = [build_hist(s, "pt", pt_bins) for s in samples]
    acop_hists = [build_hist(s, "acop", acop_bins) for s in samples]
    leading_et_hists = [build_hist(s, "leading_et", leading_et_bins) for s in samples]
    

    # Run-2 overlay (points)
    if "run2data" in data:

        def build_data_hist(variable, bins):
            obj = data["run2data"][key_map[variable]]
            return np.asarray(obj["counts"]), bins

        data_hists = [
            build_data_hist("mass", mass_bins),
            build_data_hist("pt", pt_bins),
            build_data_hist("acop", acop_bins),
            build_data_hist("leading_et", leading_et_bins),
        ]
    else:
        data_hists = [(None, None)] * 4

    labels = [
        "merged yyee " + r" $\mathrm{\gamma\gamma\rightarrow ee}$",
        "cep " + r" $\mathrm{gg\rightarrow\gamma\gamma}$",
        "signal " + r" ($\mathrm{\gamma\gamma\rightarrow\gamma\gamma}$)",
    ]
    facecolors = ["blue", "grey", "none"]
    edgecolors = ["none", "none", "red"]
    colors = ["blue", "grey", "red"]
    figsize = (10, 10)

    plot_specs = [
        (mass_hists, mass_bins, 70, "Diphoton Mass (GeV)", "diphoton_mass.png"),
        (pt_hists, pt_bins, 50, r"Diphoton $p_{T}$ (GeV)", "diphoton_pt.png"),
        (acop_hists, acop_bins, 20, r"Diphoton $A_{\phi}$", "diphoton_acop.png"),
        (
            leading_et_hists,
            leading_et_bins,
            80,
            r"Leading Photon $E_{T}$ (GeV)",
            "leading_photon_et.png",
        ),
    ]
    
    for idx, (hists, bins, ylim, xlabel, fname) in enumerate(plot_specs):
        plt.figure(figsize=figsize)
        hep.histplot(
            hists,
            histtype="fill",
            stack=True,
            linewidth=2,
            facecolor=facecolors,
            edgecolor=edgecolors,
            color=colors,
            label=labels,
            alpha=0.8,
        )
        add_error_band(hists, bins)

        # Overlay Run-2 data
        d_counts, d_bins = data_hists[idx]
        if d_counts is not None:
            centers = (d_bins[:-1] + d_bins[1:]) / 2
            errors = np.sqrt(d_counts)
            plt.errorbar(
                centers,
                d_counts,
                yerr=errors,
                fmt="o",
                markersize=6,
                color="black",
                label="Run-2 data",
            )

        add_atlas_label()
        plt.margins(x=0.05)
        plt.ylim(top=ylim)
        plt.xlabel(xlabel)
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, fname))
        plt.close()

    print("Plots written to", plot_dir)

    # ------------------------------------------------------------------
    # Persist **all** histograms (including ALPs) to a pickle
    # ------------------------------------------------------------------
    hist_pickle = {"__scale_factors__": scale_factors}

    for sample_name in data:
        hist_pickle[sample_name] = {}
        for var in variable_bins:
            counts, bins = build_hist(sample_name, var, variable_bins[var])
            values, weights = get_values_weights(sample_name, var)
            hist_pickle[sample_name][var] = {
                "counts": counts,
                "edges": bins,
                "values": values,
                "weights": weights,
            }

    out_pickle = os.path.join(plot_dir, f"histograms_{aco_mode}.pkl")
    with open(out_pickle, "wb") as fout:
        pickle.dump(hist_pickle, fout)

    print("Histograms saved to", out_pickle)
    print("Stacked plots updated.")
    # --- optional debug read-out ----------------------------------------

    totals = {s: np.sum(h[0]) for s, h in zip(samples, mass_hists)}
    print("[DEBUG] Sum of plotted events (after scaling) per sample:")
    for s, n in totals.items():
        print(f"  {s:<10}: {n:.2f}")


# ----------------------------------------------------------------------
# Cut-flow plotting (re-ordered + step-style histogram)
# ----------------------------------------------------------------------
def plot_cutflows(data, plot_dir):
    """Write a stepped cut-flow plot for every MC sample."""

    # scale_factors = compute_scale_factors(data)
    
    wanted = ("signal", "cep", "alp5")          # lbyl ≡ signal
    samples_with_cf = [s for s in wanted if s in data and "cut_flow" in data[s]] # extract the wanted sample
    # define the label
    if not samples_with_cf:
        print("No cut-flow information found - nothing to plot.")
        return

    label_base = {"signal": "lbyl", "cep": "cep", "alp5": "alp5"}

    # -- canonical order + display names --------------------------------
    ordered_steps = [
        ("All events",            "Total Events"),
        ("Photon η cut",          "Pass Photon eta cut"),
        (r"Photon $p_{T}$ cut",         "Pass Photon pt cut"),
        ("Photon mass cut",       "Pass Diphoton Selection"),  # mass ≡ diphoton sel.
        ("Charge+Pixel veto",      "Pass Pixel Track Veto"),    # use latest count
        ("Acoplanarity sel.",     "Pass Acoplanarity Selection"),
    ]

    step_labels = [lbl for lbl, _ in ordered_steps]
    x = np.arange(len(step_labels))

    # ------------------------------------------------------------------
    # Gather counts & efficiencies
    # ------------------------------------------------------------------
    counts = {}
    eff = {}
    for s in samples_with_cf:
        cnt = [
            data[s]["cut_flow"].get(key, 0)
            for _, key in ordered_steps
        ]
        counts[s] = cnt
        eff[s] = 100.0 * cnt[-1] / cnt[0] if cnt[0] > 0 else 0.0 # calculate effciency

    # -- plotting -------------------------------------------------------
    plt.figure(figsize=(8, 6))

    for idx, s in enumerate(samples_with_cf):
        y = counts[s]
        # step curve + markers
        plt.step(x, y, where="mid", linewidth=2, label=f"{label_base[s]} (ε={eff[s]:.1f}%)")
        plt.scatter(x, y, s=25)

        # annotate % removed at each step (except first)
        for j in range(1, len(y)):
            if y[j-1] > 0:
                drop = 100.0 * (y[j-1] - y[j]) / y[j-1]
                # centre position
                x_shift = x[j] + 0.06 * (idx - (len(samples_with_cf) - 1) / 2)
                # but shift the last point (Acoplanarity) a bit left
                if j == len(y) - 1:          # final bin → move left
                    x_shift -= 0.20
                plt.text(
                    x_shift,
                    y[j] * 1.1,               # a bit above the point (works with log-y)
                    f"-{drop:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
    plt.title('Cutflow efficiency for samples')
    plt.yscale("log")
    plt.ylabel("Raw events")
    plt.xticks(x, step_labels, rotation=35, ha="right")
    add_atlas_label()
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(plot_dir, "cutflow_comparison.png")
    plt.savefig(fname)
    plt.close()
    print("Cut-flow plot written to", fname)

# ----------------------------------------------------------------------
# Single‑sample plotting helper
# ----------------------------------------------------------------------

def plot_single_sample_histograms(data, sample, plot_dir, aco_mode):
    """Plot *one* sample in the same four variables as the stacked mode."""

    if sample not in data:
        print(f"Requested sample '{sample}' not available - skipping single-sample plot.")
        return

    # ------------------------------------------------------------------
    # Bin definitions – identical to `plot_histograms`
    # ------------------------------------------------------------------
    if (
        "yyee" in data
        and isinstance(data["yyee"], dict)
        and "h_ZMassZoom" in data["yyee"]
    ):
        yd = data["yyee"]
        mass_bins = np.asarray(yd["h_ZMassZoom"]["edges"])
        pt_bins = np.asarray(yd["h_ZptZoom"]["edges"])
        acop_bins = np.asarray(yd["h_ZAcoZoom"]["edges"])
        leading_et_bins = np.asarray(yd["h_ZLeadingPhotonET"]["edges"])
    else:
        mass_bins = np.linspace(0, 30, 31)
        pt_bins = np.linspace(0, 5, 51)
        acop_bins = np.linspace(0, 0.1, 101)
        leading_et_bins = np.linspace(0, 15, 31)

    variable_bins = {
        "mass": mass_bins,
        "pt": pt_bins,
        "acop": acop_bins,
        "leading_et": leading_et_bins,
    }

    # ------------------------------------------------------------------
    # Scale factors (reuse stacked helpers)
    # ------------------------------------------------------------------
    scale_factors = (
        compute_CR_scale_factors(data) if aco_mode == "cr" else compute_SR_scale_factors(data)
    )
    scale = scale_factors.get(sample, 1.0)

    key_map = {
        "mass": "h_ZMassZoom",
        "pt": "h_ZptZoom",
        "acop": "h_ZAcoZoom",
        "leading_et": "h_ZLeadingPhotonET",
    }

    def build_hist(variable, bins):
        """Return (counts, bins) for the given *single* sample."""

        # Pre‑binned samples
        if key_map[variable] in data[sample]:
            obj = data[sample][key_map[variable]]
            return np.asarray(obj["counts"]) * scale, bins

        # Array‑based samples
        res = data[sample]
        arrays = {
            "mass": res["diphoton_masses"],
            "pt": res["diphoton_pts"],
            "acop": res["diphoton_acoplanarity"],
            "leading_et": res["leading_photon_ets"],
        }
        v = np.asarray(arrays[variable])
        w = np.asarray(res.get("event_weights", np.ones_like(v))) * scale
        counts, _ = np.histogram(v, bins=bins, weights=w)
        return counts, bins

    # ------------------------------------------------------------------
    # Plotting spec
    # ------------------------------------------------------------------
    plot_specs = [
        ("mass", mass_bins, 70, "Diphoton Mass (GeV)", "diphoton_mass"),
        ("pt", pt_bins, 50, r"Diphoton $p_{T}$ (GeV)", "diphoton_pt"),
        ("acop", acop_bins, 20, r"Diphoton $A_{\phi}$", "diphoton_acop"),
        (
            "leading_et",
            leading_et_bins,
            80,
            r"Leading Photon $E_{T}$ (GeV)",
            "leading_photon_et",
        ),
    ]

    colors = {
        "yyee": "blue",
        "cep": "grey",
        "signal": "red",
        "run2data": "black",
    }
    facecolor = colors.get(sample, "green")

    for var, bins, ylim, xlabel, fname_stub in plot_specs:
        counts, _ = build_hist(var, bins)
        plt.figure(figsize=(8, 6))
        hep.histplot(
            counts,
            bins,
            histtype="fill",
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.8,
            label=sample,
        )
        add_atlas_label()
        plt.margins(x=0.05)
        plt.ylim(top=ylim)
        plt.xlabel(xlabel)
        plt.ylabel("Counts")
        plt.legend()
        out_name = os.path.join(plot_dir, f"{sample}_{fname_stub}.png")
        plt.savefig(out_name)
        plt.close()
        print(f"  - Wrote {out_name}")

    print(f"Single-sample plots for '{sample}' written to {plot_dir}.")
    # --- optional debug read-out ----------------------------------------

    # totals = {s: np.sum(h[0]) for s, h in zip(samples, mass_hists)}
    # print("[DEBUG] Sum of plotted events (after scaling) per sample:")
    # for s, n in totals.items():
    #     print(f"  {s:<10}: {n:.2f}")

# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate stacked plots, cut‑flow comparison, *or* a single‑sample histogram set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--acomode",
        choices=["sr", "cr"],
        default="sr",
        help="Acoplanarity mode: 'sr' (signal region) or 'cr' (control region).",
    )
    parser.add_argument(
        "--plot",
        choices=["stacked", "cutflow", "single"],
        default="stacked",
        help="Plot type: 'stacked' (default), 'cutflow', or 'single' sample.",
    )
    parser.add_argument(
        "--sample",
        metavar="SAMPLE",
        help="Sample label to plot when --plot single is chosen (e.g. yyee, cep, alp4, run2data).",
    )
    # keep/skip Run‑2 overlay
    parser.add_argument(
        "--no-run2",
        action="store_true",
        help="Skip loading / drawing the Run‑2 data overlay (has no effect for --plot single --sample run2data).",
    )
    parser.add_argument(
    "--debug",
    action="store_true",
    help="Print extra information such as the sum of plotted events per sample."
    )
    # ------------------------------------------------------------------
    # (SR only) acoplanarity slice selection
    # ------------------------------------------------------------------
    parser.add_argument(
        "--cut",
        default="all",
        metavar="TAG",
        help=(
            "Acoplanarity slice for SR – e.g. '0p010'.\n"
            "Use 'all' (default) to render every slice 0p005…0p020.\n"
            "Ignored for the control‑region."
        ),
    )
    args = parser.parse_args()
    global DEBUG
    DEBUG = args.debug

    # Guard: --plot single requires --sample
    if args.plot == "single" and not args.sample:
        parser.error("--plot single requires --sample to be specified.")

    # ------------------------------------------------------------------
    # Decide which acoplanarity slices to run
    # ------------------------------------------------------------------
    if args.acomode == "sr":
        cut_tags = (
            [f"0p{n:03d}" for n in range(5, 21)]  # 0p005 … 0p020
            if args.cut == "all"
            else [args.cut]
        )
    else:  # control region
        cut_tags = ["cr"]

    # Cut‑flow pickles do **not** depend on the fine slices
    cutflow_base = load_cutflow_data(args.acomode)

    for tag in cut_tags:
        # --------------------------------------------------------------
        # Build file paths for this slice / region
        # --------------------------------------------------------------
        if args.acomode == "sr":
            merged_yyee_file = (
                "/home/jtong/lbyl/yyee_binned/"
                "yyee_root_merged_hist_aco-sr.pkl"
            )
            # Run‑2 data are *not* sliced by Aco – same file for all SR cuts
            merged_run2_file = (
                "/home/jtong/lbyl/run2data_binned/"
                "run2data_root_merged_hist_aco-sr.pkl"
            )
            plot_dir = f"Plots_sr_{tag}"
        else:  # 'cr'
            merged_yyee_file = (
                "/home/jtong/lbyl/yyee_binned/"
                "yyee_root_merged_hist_aco-cr.pkl"
            )
            merged_run2_file = (
                "/home/jtong/lbyl/run2data_binned/"
                "run2data_root_merged_hist_aco-cr.pkl"
            )
            plot_dir = "Plots_cr"

        # --------------------------------------------------------------
        # Check that the required pickle(s) exist
        # --------------------------------------------------------------
        required = [merged_yyee_file]
        if not args.no_run2:
            required.append(merged_run2_file)
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            print(f"Skipping slice '{tag}' - missing: {', '.join(missing)}")
            continue

        os.makedirs(plot_dir, exist_ok=True)

        # --------------------------------------------------------------
        # Load the pickle(s)
        # --------------------------------------------------------------

        with open(merged_yyee_file, "rb") as f:
            all_slices = pickle.load(f)

        slice_key = tag if args.acomode == "sr" else "cr"
        if slice_key not in all_slices:
            print(f"Slice '{slice_key}' not found in {merged_yyee_file}. Skipping.")
            continue

        merged_yyee_data = all_slices[slice_key]

        run2_data = None
        if not args.no_run2 or (args.plot == "single" and args.sample == "run2data"):
            # The user may ask for run2data as a single sample even when --no-run2
            # is active – in that case we *must* still load it.
            if os.path.exists(merged_run2_file):
                with open(merged_run2_file, "rb") as f:
                    run2_data = pickle.load(f)
            else:
                print(f"Run-2 data pickle '{merged_run2_file}' not found - skipping run2data load.")

        # --------------------------------------------------------------
        # Assemble data dictionary for the plotting helpers
        # --------------------------------------------------------------
        data = cutflow_base.copy()  # shallow copy OK – we only read
        data["yyee"] = merged_yyee_data
        if run2_data is not None:
            data["run2data"] = run2_data

        # --------------------------------------------------------------
        # Dispatch to the chosen mode
        # --------------------------------------------------------------
        if args.plot == "cutflow":
            plot_cutflows(data, plot_dir=plot_dir)
        elif args.plot == "single":
            # NB: aco_mode=tag so output file names carry the slice label
            plot_single_sample_histograms(data, sample=args.sample, plot_dir=plot_dir, aco_mode=tag)
        else:
            # Stacked (default)
            plot_histograms(data, plot_dir=plot_dir, aco_mode=tag)


if __name__ == "__main__":
    main()
