#!/usr/bin/env python3
"""Compute fiducial cross-sections in the signal region (SR)
using the combined background & ALP pickle produced for the
LbyL analysis.

The pickle is expected to have the following top-level blocks::

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

For each signal hypothesis (SM LbyL and every ALP mass point) the script
reports:
  • signal efficiency (eff_sr)
  • σ_fid with all backgrounds subtracted:
      σ_fid = (N_data - N_yy→ee - N_CEP) / (L · eff_sr)
"""
import os
import pickle
from pprint import pprint

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def _nominal_block(sample_dict):
    """Return the dict that actually holds the histograms.

    New pickles:
        sample_dict = {"nominal": {hists…}, "systematics": {…}, …}
    Old pickles:
        sample_dict = {hists…}
    """
    return sample_dict.get("nominal", sample_dict)

def total_counts(sample_dict, hist="h_ZMassZoom"):
    """Total number of events in *hist* for *sample_dict*."""
    hists = _nominal_block(sample_dict)
    return sum(hists[hist]["counts"])


def calc_sigma(N_data, N_bkg, signal_eff, lumi):
    """Compute σ_fid given data yield, total background, efficiency and lumi."""
    if signal_eff == 0:
        raise ZeroDivisionError("Signal efficiency is zero – check input file.")
    return (N_data - N_bkg) / (lumi * signal_eff)


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------

def main(pkl_path, hist="h_ZMassZoom", lumi=1.67):
    """Entry point - calculates cross-sections using *pkl_path*."""

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)

    # Basic sanity check on the expected blocks
    required = {"lbyl", "cep", "yy2ee", "data"}
    missing = required - pkl.keys()
    if missing:
        raise KeyError(f"Missing required blocks in pickle: {missing}")

    # ------------------------------------------------------------------
    # Yields in Signal Region
    # ------------------------------------------------------------------
    N_data_SR = total_counts(pkl["data"], hist)
    N_yy2ee_SR = total_counts(pkl["yy2ee"], hist)
    N_cep_SR = total_counts(pkl["cep"], hist)
    N_bkg_SR = N_yy2ee_SR + N_cep_SR

    print("\n==================== Signal Region Yields ====================")
    print(f"Histogram used     : {hist}")
    print(f"Luminosity (nb⁻¹)  : {lumi}")
    print("------------------------------------------------------------")
    print(f"N_data_SR          = {N_data_SR}")
    print(f"N_yy→ee_SR         = {N_yy2ee_SR}")
    print(f"N_CEP_SR           = {N_cep_SR}")
    print(f"Total background   = {N_bkg_SR}\n")

    # ------------------------------------------------------------------
    # Standard LbyL signal
    # ------------------------------------------------------------------
    lbyl = pkl["lbyl"]
    eff_sr_lbyl = lbyl.get("eff_sr")
    if eff_sr_lbyl is None:                                  
        hists_lbyl = _nominal_block(lbyl)
        first_hist = next(iter(hists_lbyl.values()))
        eff_sr_lbyl = total_counts(lbyl, hist) / sum(first_hist["counts"])
        print("[INFO] eff_sr for LbyL not found - derived from histogram counts.")

    sigma_lbyl = calc_sigma(N_data_SR, N_bkg_SR, eff_sr_lbyl, lumi)

    print("==================== SM LbyL Signal =========================")
    print(f"Signal efficiency  = {eff_sr_lbyl:.6f}")
    print(f"σ_fid (nb)         = {sigma_lbyl:.4f}\n")

    # ------------------------------------------------------------------
    # ALP signals – loop over all keys starting with "alp_"
    # ------------------------------------------------------------------
    alp_results = {}
    alp_keys = [k for k in pkl if k.startswith("alp_")]

    if alp_keys:
        print("==================== ALP Signals ============================")
        for key in sorted(alp_keys, key=lambda x: float(x.split("_")[1].replace("GeV", ""))):
            alp = pkl[key]
            eff = alp["eff_sr"]
            sigma_alp = calc_sigma(N_data_SR, N_bkg_SR, eff, lumi)
            alp_results[key] = sigma_alp
            print(f"{key:<10}  eff_sr = {eff:.6f}   σ_fid = {sigma_alp:.4f} nb")
    else:
        print("[WARNING] No ALP blocks found in pickle = skipping ALP cross=sections.")

    return sigma_lbyl, alp_results


if __name__ == "__main__":
    # Update this path if your pickle lives elsewhere
    PKL_PATH = "/home/jtong/lbyl/bkg/bkg_alp_sr_pickle.pkl"
    main(PKL_PATH)
