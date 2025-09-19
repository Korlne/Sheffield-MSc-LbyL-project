#!/usr/bin/env python3
"""root_plots.py for Light-by-Light stacked/individual plots
===============================================================================
This standalone script *re-implements* **only** the histogram-plotting pieces**
from the original LbyL analysis plots.py.  It assumes the histograms are already binned
and stored in a *single* pickle file, ``bkg/bkg_pickle.pkl``, with the following
layout::

    {
        "lbyl": {
            "h_ZMassZoom":   {"bin_edges": [...], "counts": [...]},
            "h_ZptZoom":     {"bin_edges": [...], "counts": [...]},
            "h_ZAcoZoom":    {"bin_edges": [...], "counts": [...]},
            "h_ZLeadingPhotonET": {"bin_edges": [...], "counts": [...]}
        },
        "cep"   : { ... },
        "yy2ee" : { ... },        # γγ → e⁺e⁻ background
        "data"  : { ... }         # optional overlay (Run‑3)
        
        "alp_4GeV":   {                 # One block per ALP mass
            "h_ZMassZoom": { … },
            …
        },
        "alp_5GeV":   { … },
        ⋯
        "alp_100GeV": { … }
    }
    }

Each *sample* dictionary must contain the **same** four histogram keys.  The
``bin_edges`` array has length *N + 1*; the ``counts`` array length is *N*.

----------------------------------------------------------------------
Command-line quick-start
----------------------------------------------------------------------
::

    # Stacked comparison of all MC samples (+data points if present)
    $ python3 plot_histograms.py --mode stacked --pickle bkg/bkg_pickle.pkl

    # Plot *only* the CEP sample
    $ python3 plot_histograms.py --mode single --sample cep \
                                 --pickle bkg/bkg_pickle.pkl

----------------------------------------------------------------------
Matplotlib outputs
----------------------------------------------------------------------
Four PNGs will be written into the chosen output directory (default: *Plots*):

* ``diphoton_mass.png``            $m_{\gamma\gamma}$ [GeV]
* ``diphoton_pt.png``              $p_{T}^{\gamma\gamma}$ [GeV]
* ``diphoton_acop.png``            $A_{\phi}^{\gamma\gamma}$
* ``leading_photon_et.png``        $E_{T}^{\text{lead }\gamma}$ [GeV]

----------------------------------------------------------------------
Requirements
----------------------------------------------------------------------
* AnalysisBase,25.2.15

"""

# ----------------------------------------------------------------------
# Imports – keep minimal & document purpose
# ----------------------------------------------------------------------
import argparse                           # CLI argument handling
import os                                 # Filesystem paths / mkdir
import pickle                             # Read the histogram pickle

import numpy as np                        # Array maths & histograms
import matplotlib.pyplot as plt           # Core plotting library
import mplhep as hep                      # ATLAS plot style utilities

# Apply ATLAS‑like style
hep.style.use(hep.style.ATLASAlt)

# ----------------------------------------------------------------------
# Constants – mapping variable → pickle key & axis labels
# ----------------------------------------------------------------------
VARIABLES = {
    "mass":       ("h_ZMassZoom",         r"Diphoton Mass $m_{\gamma\gamma}$ [GeV]",   (0,25)),
    "pt":         ("h_ZptZoom",           r"Diphoton $p_{T}^{\gamma\gamma}$ [GeV]",   (0,30)),
    "acop":       ("h_ZAcoZoom",          r"Diphoton $A_{\phi}^{\gamma\gamma}$",      (0,30)),
    "leading_et": ("h_ZLeadingPhotonET",  r"Leading Photon $E_{T}$ [GeV]",              (0,30)),
}

# Order in which MC samples are stacked (front → back)
STACK_ORDER = ["yy2ee", "cep", "lbyl"]  # customise if needed

# ----------------------------------------------------------------------
# Helper: read histogram arrays from the pickle
# ----------------------------------------------------------------------

def get_hist(sample_dict, hist_key):
    """Return (counts, edges) for the *nominal* histogram of a sample.

    Works with both the new structure
        sample_dict = {"nominal": {...}, "systematics": {...}, ...}
    """
    # If the new layout is present, dive into the "nominal" block
    sample_dict = sample_dict.get("nominal", sample_dict)

    h = sample_dict[hist_key]
    return np.asarray(h["counts"]), np.asarray(h["bin_edges"])

# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------

def add_atlas_label(com=5.36):
    hep.atlas.label(
        label="",
        loc=4,
        # modify the sr or cr label before use
        rlabel=r"Pb+Pb $\sqrt{s_{NN}} = " + f"{com}" + r"\,\mathrm{TeV}$" + ", 1.63 nb$^{-1}$" + "\nSignal Region",
        year=2025,
    )

def plot_stacked(data, outdir, acop_xlim=(0, 0.01), nodata=False):
    """Create four stacked MC plots (+optional data overlay)."""

    os.makedirs(outdir, exist_ok=True)

    # Colours chosen to match original script – adapt if desired
    facecolors = {"yy2ee": "tab:blue", "cep": "lightgrey", "lbyl": "none"}
    edgecolors = {"yy2ee": "none",     "cep": "none",      "lbyl": "tab:red"}

    for var, (key, xlabel, ylim) in VARIABLES.items():
        # Build list of (counts, edges) in the specified stack order
        hists = [get_hist(data[s], key) for s in STACK_ORDER if s in data]

        # MC stack
        plt.figure(figsize=(8, 6))
        hep.histplot(
            hists,
            histtype="fill",
            stack=True,
            linewidth=1.2,
            facecolor=[facecolors[s] for s in STACK_ORDER if s in data],
            edgecolor=[edgecolors[s] for s in STACK_ORDER if s in data],
            label=[s for s in STACK_ORDER if s in data],
            alpha=0.8,
        )

        # 1 σ stat‑error band of the *total* stacked MC
        total_counts = np.sum([h[0] for h in hists], axis=0)
        total_edges  = hists[0][1]
        centres = 0.5 * (total_edges[:-1] + total_edges[1:])
        centres       = np.insert(centres, 0, total_edges[0])    # x = [edge_0 , centres …]
        total_counts  = np.insert(total_counts, 0, total_counts[0])
        plt.fill_between(
            centres,
            total_counts - np.sqrt(total_counts),
            total_counts + np.sqrt(total_counts),
            step="mid",
            color="none",
            hatch="/////",
            edgecolor="black",
            linewidth=0,
            label="Stat. unc.",
        )

        # Optional Run-3 data overlay (suppressed if --nodata)
        if (not nodata) and "data" in data:
            d_counts, d_edges = get_hist(data["data"], key)
            centres_d = 0.5 * (d_edges[:-1] + d_edges[1:])
            plt.errorbar(
                centres_d, d_counts, yerr=np.sqrt(d_counts), fmt="ko", markersize=4,
                label="Run-3 data",
            )

        # hep.atlas.label("Work in Progress", loc=1, year=2025)
        add_atlas_label()
        plt.xlabel(xlabel, labelpad=20)
        plt.ylabel("Events")
        plt.ylim(ylim)
        plt.margins(x=0.05)
        if var == "acop":
            plt.xlim(*acop_xlim)  # so aco plot is readable
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(outdir, f"{var}.png")
        plt.savefig(fname)
        plt.close()
        print(f"[stacked] Wrote {fname}")


def plot_single(data, sample, outdir, acop_xlim=(0, 0.01)):
    """Plot the four variables for *one* sample (filled histograms)."""

    if sample not in data:
        raise ValueError(f"Sample '{sample}' not found in pickle.")

    os.makedirs(outdir, exist_ok=True)

    colour = "tab:red" if sample == "lbyl" else "tab:blue"

    for var, (key, xlabel, ylim) in VARIABLES.items():
        counts, edges = get_hist(data[sample], key)
        plt.figure(figsize=(8, 6))
        hep.histplot(
            counts,
            edges,
            histtype="fill",
            facecolor=colour,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.8,
            label=sample,
        )
        
        # hep.atlas.label("Work in Progress", loc=1, year=2025)
        add_atlas_label()
        plt.xlabel(xlabel, labelpad=10)
        plt.ylabel("Events")
        plt.ylim(ylim)
        plt.margins(x=0.05)
        if var == "acop":
            plt.xlim(*acop_xlim)  # so aco plot is readable
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(outdir, f"{sample}_{var}.png")
        plt.savefig(fname)
        plt.close()
        print(f"[single]  Wrote {fname}")

# ----------------------------------------------------------------------
# Main entry point: parse CLI and dispatch
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot stacked MC histograms or a single sample from a LbyL pickle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pickle", default="bkg/bkg_alp_sr_pickle.pkl", help="Path to histogram pickle file.")
    ap.add_argument("--outdir", default="Plots/root_hist_sr", help="Output directory for PNGs.")
    ap.add_argument("--plotmode", choices=["stacked", "single"], default="stacked",
                    help="Plot *all* MC in a stack or a single sample.")
    ap.add_argument("--sample", help="Sample name for --mode single (e.g. lbyl, cep, yy2ee, data, alp4,5,6...).")

    # ---- control-region plotting ---------------------------------
    ap.add_argument(
        "--crpickle",
        default="/home/jtong/lbyl/bkg/bkg_cr_pickle.pkl",
        help="Path to control-region histogram pickle file (plots go to --outdir_cr).",
    )
    ap.add_argument(
        "--outdir_cr",
        default="Plots/root_hist_cr",
        help="Output directory for control-region PNGs.",
    )
    ap.add_argument("--nodata", action="store_true",
        help="Suppress Run-3 data overlay in stacked plots.")
    
    args = ap.parse_args()
    # ------------------------------------------------------------------
    # Load pickle
    # ------------------------------------------------------------------
    if not os.path.exists(args.pickle):
        raise FileNotFoundError(f"Pickle '{args.pickle}' not found.")
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    # ------------------------------------------------------------------
    # Route to chosen plotting mode
    # ------------------------------------------------------------------
    if args.plotmode == "stacked":
        plot_stacked(data, args.outdir, nodata=args.nodata)
    else:  # single
        if not args.sample:
            raise SystemExit("--mode single requires --sample <name> to be specified.")
        plot_single(data, args.sample, args.outdir)

    # ------------------------------------------------------------------
    # (Optional) ALSO plot the control-region histograms
    # ------------------------------------------------------------------
    if args.crpickle and os.path.exists(args.crpickle):
        with open(args.crpickle, "rb") as f:
            data_cr = pickle.load(f)

        print(f"\n[info] Found CR pickle → plotting into '{args.outdir_cr}'")
        if args.plotmode == "stacked":
            plot_stacked(data_cr, args.outdir_cr, acop_xlim=(0, 0.05), nodata=args.nodata)
        else:
            if not args.sample:
                raise SystemExit("--mode single requires --sample <name> to be specified.")
            plot_single(data_cr, args.sample, args.outdir_cr, acop_xlim=(0, 0.1))

if __name__ == "__main__":
    main()
