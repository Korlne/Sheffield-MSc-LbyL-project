#!/usr/bin/env python
"""
Produce histograms for LbyL analysis **using per‑event weights** so that only the
bin *heights* are scaled.  This script supersedes the original ``plots.py``.

Key changes
-----------
* For the MC samples (``signal`` and ``cep``) the script now pulls the
  ``event_weights`` array from the cut‑flow pickle and passes it as the
  ``weights=`` argument to ``np.histogram``.
* Additional global scale factors (luminosity×cross‑section, control‑region
  transfer factors, …) are folded into the weight array **before** the
  histogram call, so the binning itself remains untouched.
* Fixed the leading‑photon E_T source (was rapidity diff by mistake).

Usage (unchanged)
-----------------
$ python plots_weighted.py            # default signal‑region (sr)
$ python plots_weighted.py --acomode cr
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from AthenaCommon.SystemOfUnits import nanobarn

hep.style.use(hep.style.ATLASAlt)

# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def load_cutflow_data(aco_mode):
    """Return a dict keyed by sample name with the unpickled results."""
    samples = ["yyee", "cep", "signal"]
    out = {}
    for s in samples:
        fn = f"cutflow_{s}_aco-{aco_mode}.pkl"
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                out[s] = pickle.load(f)
        else:
            print(f"[WARN] Missing cut‑flow file {fn}; skipping {s}.")
    return out

# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def add_atlas_label(com=5.36):
    hep.atlas.label(label="Work in Progress", loc=4,
                    rlabel=fr"Pb+Pb $\sqrt{{s_{{NN}}}}={com}\,\mathrm{{TeV}}$", year=2025)


def add_error_band(hist_list, bins, hatch="xxxxx"):
    """Overlay a simple √N error band for stacked histograms."""
    counts = np.sum([h[0] for h in hist_list], axis=0)
    errs = np.sqrt(counts)
    centers = 0.5 * (bins[:-1] + bins[1:])
    plt.fill_between(centers, counts - errs, counts + errs, step="mid",
                     facecolor="none", edgecolor="k", hatch=hatch, alpha=0.5,
                     linewidth=0, label="Syst. uncertainty")

# ----------------------------------------------------------------------
# Main plotting routine
# ----------------------------------------------------------------------

def plot_histograms(data, plot_dir):
    """Build and save all histograms for the three samples."""

    # ------------------------------------------------------------------
    # Bin definitions – take from merged yyee if available
    # ------------------------------------------------------------------
    if "yyee" in data and isinstance(data["yyee"], dict) and "h_ZMassZoom" in data["yyee"]:
        yd = data["yyee"]
        mass_bins       = np.asarray(yd["h_ZMassZoom"]["edges"])
        pt_bins         = np.asarray(yd["h_ZptZoom"]["edges"])
        acop_bins       = np.asarray(yd["h_ZAcoZoom"]["edges"])
        leading_et_bins = np.asarray(yd["h_ZLeadingPhotonET"]["edges"])
    else:
        mass_bins       = np.linspace(0, 30, 31)
        pt_bins         = np.linspace(0, 5, 51)
        acop_bins       = np.linspace(0, 0.1, 101)
        leading_et_bins = np.linspace(0, 15, 31)

    costheta_bins      = np.linspace(0, 1, 11)
    rapidity_bins      = np.linspace(-5, 5, 31)

    # ------------------------------------------------------------------
    # Global scale factors
    # ------------------------------------------------------------------
    total_signal_events = 100000.0  # MC statistics normalisation
    data_lumi           = 1.67 / nanobarn
    cross_section       = 879.2621 * nanobarn
    signal_scale        = data_lumi * cross_section / total_signal_events

    # Control‑region normalisations (hard‑coded as in original script)
    N_data_CR   = 219
    N_yyee_CR   = 70
    N_sig_CR    = data["signal"]["cut_flow"]["Pass Acoplanarity Selection"]
    N_cep_MC_CR = data["cep"]["cut_flow"]["Pass Acoplanarity Selection"]
    yd          = data["yyee"]
    N_yyee_MC_CR = np.sum(yd["h_ZMassZoom"]["counts"])

    N_cep_data  = N_data_CR - N_yyee_CR - (N_sig_CR * signal_scale)

    cep_scale   = N_cep_data / N_cep_MC_CR if N_cep_MC_CR else 1.0
    yyee_scale  = N_yyee_CR / N_yyee_MC_CR if N_yyee_MC_CR else 1.0

    scale_factors = {"yyee": yyee_scale, "cep": cep_scale, "signal": signal_scale}

    # ------------------------------------------------------------------
    # Build histograms – now weight‑aware for cep & signal
    # ------------------------------------------------------------------
    def build_hist(sample, variable, bins):
        """Return (counts, bins) for given variable list."""
        scale = scale_factors[sample]

        if sample == "yyee":
            # Pre‑merged counts – just scale
            key_map = {
                "mass": "h_ZMassZoom",
                "pt": "h_ZptZoom",
                "acop": "h_ZAcoZoom",
                "leading_et": "h_ZLeadingPhotonET",
            }
            obj = data["yyee"][key_map[variable]]
            return (np.asarray(obj["counts"]) * scale, bins)

        # MC samples with per‑event arrays + weights
        results = data[sample]
        arrays = {
            "mass": results["diphoton_masses"],
            "pt":   results["diphoton_pts"],
            "acop": results["diphoton_acoplanarity"],
            "leading_et": results["leading_photon_ets"],
            "rapidity": results["diphoton_rapidity_diff"],
            "costheta": results["diphoton_cos_thetas"],
        }
        vals = np.asarray(arrays[variable])
        w    = np.asarray(results.get("event_weights", np.ones_like(vals))) * scale
        counts, _ = np.histogram(vals, bins=bins, weights=w)
        return (counts, bins)

    # Histograms per sample
    mass_hists       = [build_hist(s, "mass", mass_bins)        for s in ("yyee", "cep", "signal")]
    pt_hists         = [build_hist(s, "pt",   pt_bins)          for s in ("yyee", "cep", "signal")]
    acop_hists       = [build_hist(s, "acop", acop_bins)        for s in ("yyee", "cep", "signal")]
    leading_et_hists = [build_hist(s, "leading_et", leading_et_bins) for s in ("yyee", "cep", "signal")]
    rapidity_hists   = [build_hist(s, "rapidity", rapidity_bins)    for s in ("yyee", "cep", "signal")]
    costheta_hists   = [build_hist(s, "costheta", costheta_bins)    for s in ("yyee", "cep", "signal")]

    labels = [
        "merged yyee " + r' $\mathrm{\gamma\gamma\rightarrow ee}$',
        "cep " + r' $\mathrm{gg\rightarrow\gamma\gamma}$',
        "signal " + r' ($\mathrm{\gamma\gamma\rightarrow\gamma\gamma}$)',
    ]
    facecolors = ["blue", "grey", "none"]
    edgecolors = ["none", "none", "red"]
    colors     = ["blue", "grey", "red"]
    figsize    = (10, 10)

    def save(name):
        plt.savefig(os.path.join(plot_dir, name))
        plt.close()

    # ---------------- Plots ----------------
    plots = [
        (mass_hists, mass_bins, 60, "Diphoton Mass (GeV)", "diphoton_mass.png"),
        (pt_hists,   pt_bins,   40, r"Diphoton $p_{T}$ (GeV)", "diphoton_pt.png"),
        (acop_hists, acop_bins, 30, r"Diphoton $A_{\phi}$", "diphoton_acop.png"),
        (leading_et_hists, leading_et_bins, 50, r"Leading Photon $E_{T}$ (GeV)", "leading_photon_et.png"),
        (rapidity_hists, rapidity_bins, 300, r"$\Delta y$", "diphoton_rapidity.png"),
    ]

    for hists, bins, ylim, xlabel, fname in plots:
        plt.figure(figsize=figsize)
        hep.histplot(hists, histtype="fill", stack=True, linewidth=2,
                     facecolor=facecolors, edgecolor=edgecolors, color=colors,
                     label=labels, alpha=0.8)
        add_error_band(hists, bins)
        add_atlas_label()
        plt.margins(x=0.05)
        plt.ylim(top=ylim)
        plt.xlabel(xlabel)
        plt.ylabel("Counts")
        plt.legend()
        save(fname)

    print("Plots written to", plot_dir)
    print("Plots updated.")

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate weighted plots for LbyL analysis")
    ap.add_argument("--acomode", choices=["sr", "cr"], default="sr",
                    help="Acoplanarity mode: 'sr' (signal) or 'cr' (control) region")
    args = ap.parse_args()

    os.makedirs("Plots", exist_ok=True)

    merged_file = "/home/jtong/lbyl/yyee_binned/yyee_root_merged_hist.pkl"
    if not os.path.exists(merged_file):
        raise FileNotFoundError(f"Merged yyee histogram file {merged_file} not found")
    with open(merged_file, "rb") as f:
        merged_yyee = pickle.load(f)

    data = load_cutflow_data(args.acomode)
    data["yyee"] = merged_yyee  # override with merged histos

    out_dir = f"Plots_{args.acomode}"
    os.makedirs(out_dir, exist_ok=True)
    plot_histograms(data, out_dir)

if __name__ == "__main__":
    main()
