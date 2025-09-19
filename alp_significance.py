#!/usr/bin/env python3
import pickle
import numpy as np
import os

# ------------- Config -------------------------------------------------------
LUMI_NB = 1.73                     # integrated luminosity in nb⁻¹ 
CUTFLOW_STAGE = "Pass Diphoton Selection"   # key used to extract the SR yield
UNCERTAINTY_FRAC = 0.20            # 20 % systematic on total background

# Background pickles
BG_PICKLES = {
    "lbl": "cutflow_signal_aco-sr_scan.pkl",  # light by light
    "cep": "cutflow_cep_aco-sr_scan.pkl",
}

# Signal pickles (ALP masses)
SIGNAL_PICKLES = {
    5: "cutflow_alp5_aco-sr_scan.pkl",
    6: "cutflow_alp6_aco-sr_scan.pkl",
}

print("\n--- ALP significance (SR-scan) ---")

# ------------- 1) Load background ------------------------------------------
bkg_total = 0.0
for tag, pkl in BG_PICKLES.items():
    with open(pkl, "rb") as f:
        cf = pickle.load(f)
        y = cf.get(CUTFLOW_STAGE, 0)
        print(f"{tag.upper():5s} SR yield: {y:.2f}")
        bkg_total += y

print(f"Total background B: {bkg_total:.2f}\n")

# ------------- 2) Loop over ALP masses -------------------------------------
print(f"{'m_ALP (GeV)':<12} {'S':<8} {'Z=S/sqrt(B)':<15} {'Z (w/ δB)':<15}")
print("-" * 50)

for mass, pkl in SIGNAL_PICKLES.items():
    if not os.path.exists(pkl):
        print(f"{mass:<12} MISSING FILE")
        continue

    with open(pkl, "rb") as f:
        cf = pickle.load(f)
        s = cf.get(CUTFLOW_STAGE, 0)

    z_nominal = s / np.sqrt(bkg_total) if bkg_total else 0
    z_syst    = s / np.sqrt(bkg_total + (UNCERTAINTY_FRAC * bkg_total) ** 2) if bkg_total else 0

    print(f"{mass:<12} {s:<8.2f} {z_nominal:<15.2f} {z_syst:<15.2f}")
