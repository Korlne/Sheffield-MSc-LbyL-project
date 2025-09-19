#!/usr/bin/env python
"""Process YY→ee ROOT samples (Signal or Control region) and pickle histograms.

This script now supports the optional command-line flag ``--aco`` which takes
``sr`` (signal region) or ``cr`` (control region).  Depending on the chosen
region the appropriate set of ROOT files is processed, and all intermediate
pickle files as well as the merged output are tagged with the region to avoid
confusion.

Only the minimal plumbing required for the new option has been added; the
original coding style and function layout are preserved.

all_merged                                # top-level object that is pickled
│
├── "<slice-key 1>"                      # str
│     ├── "h_ZMassZoom"                  # histogram name
│     │     ├── "edges"   → list[float]  # length = nBins+1 (bin edges)
│     │     ├── "counts"  → list[float]  # length = nBins     (scaled bin contents)
│     │     └── "errors"  → list[float]  # length = nBins     (scaled bin errors)
│     ├── "h_ZptZoom"
│     ├── "h_Zrapidity"
│     ├── "h_ZAcoZoom"
│     └── "h_ZLeadingPhotonET"
│
├── "<slice-key 2>"
│     └── …                              # identical structure
│
└── … (one entry per acoplanarity slice)
"""

import argparse
import os
import glob
import pickle

import numpy as np
import ROOT
from AthenaCommon.SystemOfUnits import nanobarn

###############################################################################
# Command‑line handling                                                       #
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract, scale and pickle post-AcoCut histograms from YY→ee"
                    " samples in either the signal (sr) or control (cr) region."
    )
    parser.add_argument(
        "--cut",
        default="all",
        help="Acoplanarity slice for SR (e.g. '0p010'). "
             "Use 'all' (default) to process every SR slice.",
    )
    parser.add_argument(
        "--aco",
        required=True,
        choices=["sr", "cr"],
        help="Select which YY→ee files to process: 'sr' (signal region) or 'cr'\n"
             "(control region)."
    )
    return parser.parse_args()

###############################################################################
# File lists                                                                  #
###############################################################################

# NOTE:  The control‑region (cr) file list is currently a placeholder – add the
# real paths when they become available.
ROOT_FILES = {
    # Control‑region (cr) files – the original sample list from Lucy.
    "cr": [
        "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860189.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611856_00_198765_Smc5000_Sppc2500_20M.root",
        "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860188.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611840_00_198765_Smc5000_Sppc2500_7M20.root",
        "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860187.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611888_00_198765_Smc5000_Sppc2500_4M7.root",
        "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860222.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid44229239_00_198765_Smc5000_Sppc2500_3M4.root",
    ],

    # Signal‑region (sr) files – the original sample list from Lucy.
    "sr": [
         "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860189.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611856_00_1987659901100_Smc5000_Sppc2500_20M.root",
         "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860188.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611840_00_1987659901100_Smc5000_Sppc2500_7M20.root",
         "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860187.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611888_00_1987659901100_Smc5000_Sppc2500_4M7.root",
         "/home/jtong/lbyl/yyee_binned/user.slawlor.Lbylntuples.860222.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid44229239_00_1987659901100_Smc5000_Sppc2500_3M4.root",
    ],
}

###############################################################################
# Analysis configuration                                                      #
###############################################################################

# Directory under 'Nominal' containing the post‑AcoCut histograms
# ------------------------------------------------------------------
# Control-region keeps the old single directory, but the SR now has
# one directory per acoplanarity slice:              m_3LbLeventselection_<CUT>
# where <CUT> runs from 0p005 to 0p020.  Keep a helper list.
# ------------------------------------------------------------------
HIST_DIR_KEY = "3_LbLeventselection"          # for CR / legacy paths
SR_CUTS = [f"0p{n:03d}" for n in range(5, 21)]    # 0p005 … 0p020

# Names of histograms to extract
HIST_NAMES = [
    "h_ZMassZoom",
    "h_ZptZoom",
    "h_Zrapidity",
    "h_ZAcoZoom",
    "h_ZLeadingPhotonET",
]

# Cross‑section map (in nanobarn) keyed by sample token
CROSS_SECTION = {
    "20M": 7780 * nanobarn,
    "7M20": 210000 * nanobarn,
    "4M7": 750000 * nanobarn,
    "3M4": 938097 * nanobarn,
}

###############################################################################
# Helper functions                                                            #
###############################################################################

def extract_histogram_info(hist):
    """Return (edges, counts, errors) as NumPy arrays from a ROOT TH1."""
    n_bins = hist.GetNbinsX()
    edges = [hist.GetBinLowEdge(i) for i in range(1, n_bins + 1)]
    edges.append(hist.GetBinLowEdge(n_bins) + hist.GetBinWidth(n_bins))
    counts = [hist.GetBinContent(i) for i in range(1, n_bins + 1)]
    errors = [hist.GetBinError(i) for i in range(1, n_bins + 1)]
    return np.array(edges), np.array(counts), np.array(errors)


def process_root_files(files, region, hist_dir_key, cut_label=None):
    """Extract, scale and pickle histograms for the given region."""
    data_lumi = 1.67 / nanobarn
    total_mc_ev = 0  # Accumulate total MC events for bookkeeping only

    for filepath in files:
        basename = os.path.basename(filepath)
        
        # --- FIX: Correctly extract the token from the end of the filename ---
        # The original `basename.split(".")[3]` was incorrect.
        # This new method splits the filename by the last underscore "_" and
        # then removes the ".root" extension to get the correct token (e.g., "3M4").
        token = basename.rsplit("_", 1)[-1].split(".")[0]
        
        print(f"[{region.upper()}] Processing {basename} (token={token})")

        f = ROOT.TFile.Open(filepath)
        if not f or f.IsZombie():
            print("  ERROR: cannot open ROOT file - skipping")
            continue

        prefix = f"Nominal/{hist_dir_key}/Pair"
        hist_info = {}

        # Extract requested histograms
        for name in HIST_NAMES:
            h = f.Get(f"{prefix}/{name}")
            if not h:
                print(f"  WARNING: {name} not found in {prefix}")
                continue
            edges, counts, errors = extract_histogram_info(h)
            hist_info[name] = {
                "edges": edges,
                "counts": counts,
                "errors": errors,
            }

        # Determine scaling factor from total MC events
        tot = f.Get("eventVeto/TotalEvents")
        mc_ev = tot.GetBinContent(2) if tot else 0
        cross = CROSS_SECTION.get(token, 1.0)
        scale = (data_lumi * cross / mc_ev) if mc_ev else 1.0
        print(f"  Scale factor = {scale}")
        total_mc_ev += mc_ev
        print(f"  mc_ev = {mc_ev:,.0f},   token = {token},   scale = {scale:.3g}")

        # Apply scaling
        for d in hist_info.values():
            d["counts"] *= scale
            d["errors"] *= scale
        # print('value of scaled counts = ', d["counts"])
        # print('sum of scaled counts = ', np.sum(d["counts"]))
        # Write pickle with region tag
        lbl = cut_label or "base"
        outname = os.path.join(
            os.path.dirname(filepath),
            f"yyee_root_{token}_{region}_aco-{lbl}_hist.pkl",
        )
        with open(outname, "wb") as fo:
            pickle.dump(hist_info, fo)
        print(f"  Saved {outname}\n")

    print(f"→ Total MC events processed ({region.upper()}): {total_mc_ev}")


def merge_histograms(pkl_dir, hist_names, region, cut_label=None):
    """Merge all per-file pickles matching *region* and (optional) *cut*."""
    clbl = cut_label or "*"
    pattern = os.path.join(pkl_dir, f"yyee_root_*_{region}_aco-{clbl}_hist.pkl")
    files = [pf for pf in glob.glob(pattern) if "merged" not in pf]
    merged = {name: None for name in hist_names}

    for pf in files:
        data = pickle.load(open(pf, "rb"))
        for name in hist_names:
            d = data.get(name)
            if d is None:
                continue
            edges = np.array(d["edges"])
            counts = np.array(d["counts"], float)
            errors = np.array(d["errors"], float)

            if merged[name] is None:
                merged[name] = {
                    "edges": edges.copy(),
                    "counts": counts.copy(),
                    "errors": errors.copy(),
                }
            else:
                merged[name]["counts"] += counts
                merged[name]["errors"] = np.sqrt(
                    merged[name]["errors"] ** 2 + errors ** 2
                )

    return merged

###############################################################################
# Main driver                                                                 #
###############################################################################

if __name__ == "__main__":
    args   = parse_args()
    region = args.aco.lower()

    root_files = ROOT_FILES.get(region)
    if not root_files:
        raise RuntimeError(f"No ROOT files configured for region '{region}'.")

    # -----------------------------------------------------------------
    # Decide which acoplanarity slices to run (CR still has a single one)
    # -----------------------------------------------------------------
    if region == "sr":
        cuts = SR_CUTS if args.cut == "all" else [args.cut]
    else:
        cuts = [None]                        # CR → single pass

    pkl_dir      = os.path.dirname(root_files[0])
    all_merged   = {}                       # ← NEW container
    overall_evts = 0                        # keep a grand total

    for cut in cuts:
        hist_dir = HIST_DIR_KEY if region == "cr" else f"m_3LbLeventselection_{cut}"

        # 1) Per-file pickles (unchanged)
        process_root_files(root_files, region, hist_dir, cut)

        # 2) Merge them (unchanged)
        merged = merge_histograms(pkl_dir, HIST_NAMES, region, cut)

        # Stash under its own key (use slice label or 'cr')
        key              = cut or "cr"
        all_merged[key]  = merged

        # Book-keeping for a short summary later
        overall_evts    += sum(
            sum(merged[name]["counts"]) for name in HIST_NAMES
        )

    # -----------------------------------------------------------------
    # One output file holding every slice
    # -----------------------------------------------------------------
    for merged in all_merged.values():                  # cast NumPy → list
        for d in merged.values():
            d["edges"]  = d["edges"].tolist()
            d["counts"] = d["counts"].tolist()
            d["errors"] = d["errors"].tolist()

    merged_out = os.path.join(
        pkl_dir, f"yyee_root_merged_hist_aco-{region}.pkl"
    )
    with open(merged_out, "wb") as fo:
        pickle.dump(all_merged, fo)
    print(f"Saved merged pickle with {len(all_merged)} slice(s): {merged_out}")

    # -- short summary ---------------------------------------------------------
    print(f"\n=== Grand totals ({region.upper()}) ===")
    for slice_key, merged in all_merged.items():
        total = sum(merged["h_ZMassZoom"]["counts"])
        print(f"  slice '{slice_key}': {total:,.1f} weighted events")
