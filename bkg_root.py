"""
Histogram normalisation & pickling for LbyL background interpretation
====================================================================

Purpose
-------
* Scan ROOT files under /home/jtong/lbyl/bkg/backgrounds_interpretation
* Extract five histograms per sample from the folder

      Nominal/3_LbLeventselection/Pair/<hist_name>

* Merge histograms that belong to the same physics sample (e.g. four
  yy→ee files become one set of histograms).
  - Standard samples:  lbyl, CEP, γγ→ee  
  - ALP samples:       one sample per mass point (folder name ``<mass>GeV``).

* Scale the Monte-Carlo samples so their **signal-region** integrals match
  the expected numbers

      lbyl  → 51.68  events  
      CEP   → 13.54  events  
      yy→ee → 14.38  events

  For ALP samples the expected yield is
  
  N_exp = (σ*L)/10,000
  
  where the _10 000_ denominator corresponds to the number of generated
  Monte-Carlo events per mass point.

* Save everything to a pickle called **bkg_pickle**.

Pickle layout
-------------
The file you get back is a plain Python dictionary:

    {
        "lbyl": {                     # Standard samples …
            "h_ZMassZoom": {
                "bin_edges": [e0, e1, …, eN],   # length N+1
                "counts"   : [c0, c1, …, c_{N-1}]   # length N
            },
            "h_ZptZoom":   { … },
            …
        },
        "cep":      { … },
        "yy2ee":    { … },
        "data":     { … },

        "alp_4GeV":   {                 # One block per ALP mass
            "h_ZMassZoom": { … },
            …
            "eff_sr":xxx ,
        },
        "alp_5GeV":   { … },
        ⋯
        "alp_100GeV": { … }
    }

Load it later like so::

    import pickle, matplotlib.pyplot as plt
    with open("bkg_pickle", "rb") as f:
        h = pickle.load(f)

    y = h["lbyl"]["h_ZptZoom"]["counts"]
    x = h["lbyl"]["h_ZptZoom"]["bin_edges"]
    plt.step(x[:-1], y, where="post"); plt.show()

All numbers are floating-point; integrals of MC histograms equal the
expected event yields after scaling.
"""

import os
import numpy as np
import pickle
from collections import defaultdict
import uproot

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

BASE_DIR = "/home/jtong/lbyl/bkg/backgrounds_interpretation"
CR_BASE_DIR = "/home/jtong/lbyl/bkg/cr"
ALP_BASE_DIR = "/home/jtong/lbyl/data/alp/ALP"
HIST_DIR_KEY_CR  = "Nominal/Grl/Trigger/TopoClusterCut/NMstEq0/NBlSpacePointEq0/NForwardElectronsAny/NGoodPhotonsGt0/NGoodPhotonsEq2/NPidPhotonsEq2/InvariantMassGt5GeV/DiPhotonPtLt1GeV/NTrkEq0_NPixTrkEq0/InvAcoCut/Pair"

# List of histogram names to extract



# List of histogram names to extract
HIST_NAMES = [
    "h_ZMassZoom",
    "h_ZMassFine",
    "h_ZptZoom",
    "h_Zrapidity",
    "h_ZAcoZoom",
    "h_ZLeadingPhotonET",
]

# Mapping of samples to ROOT files
SAMPLES = {
    "lbyl": ["LbLsignal.root"],
    "cep": ["cepincoh.root"],
    "yy2ee": [
        "3M4highstat.root",
        "4M7.root",
        "7M20.root",
        "M20.root",
    ],
    "data": ["data.root"],  # Data sample (no normalisation applied)
}

# --------------------------------------------------------------------
# === CR: file lists =================================================
# --------------------------------------------------------------------
CR_SAMPLES = {
    "lbyl": [
        "user.llewitt.Lbylntuples.860223.AOD.e8558_e8528_s4357_r15716_r14663_tid40981863_00_2028_Smc5000_Sppc2500_LbyL.root",
    ],
    "cep": [
        "user.llewitt.Lbylntuples.860225.AOD.e8558_e8528_s4357_s4258_r15716_r14663_tid41014010_00_2028_Smc5000_Sppc2500_incohCEP.root",
    ],
    "yy2ee": [
        "user.llewitt.Lbylntuples.860187.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611888_00_2028_Smc5000_Sppc2500_4M7yyee.root",
        "user.llewitt.Lbylntuples.860188.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611840_00_2028_Smc5000_Sppc2500_7M20yyee.root",
        "user.llewitt.Lbylntuples.860189.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid41611856_00_2028_Smc5000_Sppc2500_20Myyee.root",
        "user.llewitt.Lbylntuples.860222.AOD.e8587_e8586_s4357_s4258_r15716_r14663_tid44229239_00_2028_Smc5000_Sppc2500_3M4yyeehighstats.root",
    ],
    "data": [
        "user.llewitt.Lbylntuples.periodAllYear2.DAOD_HION4.grp23_v01_p6774_2028_Smc5000_Sppc2500.root",
    ],
}

# Expected number of signal‑region events used to normalise MC samples
EXPECTED_COUNTS = {
    "lbyl": 51.68,
    "cep": 13.54,
    "yy2ee": 14.38,
    # Data is actual data – no expected scaling
}

# Expected number of control-region events used to normalise MC samples
EXPECTED_CR_COUNTS = {
    "lbyl": 11.01,
    "cep": 51.67,
    "yy2ee": 12.32,
    # Data is actual data – no expected scaling
}

TOTAL_EVENTS_KEY = "eventVeto/TotalEvents"  # one‑bin histo → integral == N_total


def get_total_events(root_file):
    """Return the total number of generated events in an ALP file from
    ``eventVeto/TotalEvents`` (one-bin histogram → integral ≡ N_total)."""
    try:
        counts, _ = root_file[TOTAL_EVENTS_KEY].to_numpy()
        return float(counts.sum())
    except Exception as err:
        raise RuntimeError(
            f"Cannot read total events from '{TOTAL_EVENTS_KEY}' ({err})"
        )

# --------------------------------------------------------------------
# === ALP support ====================================================
# --------------------------------------------------------------------
from AthenaCommon.SystemOfUnits import nanobarn

alp_sigma_nb = {
    4: 7.967330e3, 5: 6.953744e3, 6: 6.044791e3, 7: 5.300250e3,
    8: 4.670220e3, 9: 4.154600e3, 10: 3.709976e3, 12: 3.016039e3,
    14: 2.499097e3, 15: 2.285133e3, 16: 2.093761e3, 18: 1.782345e3,
    20: 1.526278e3, 30: 7.779030e2, 40: 4.368360e2, 50: 2.600118e2,
    60: 1.604056e2, 70: 1.016849e2, 80: 6.546058e1, 90: 4.280824e1,
    100: 2.824225e1,
}

# Integrated luminosity (≈ 1.67 nb⁻¹)
dataLumi = 1.63 / nanobarn

# Dynamically extend SAMPLES + EXPECTED_COUNTS
alp_expected = {}
alp_sample_map = {}

for mass, sigma_nb in alp_sigma_nb.items():
    tag = f"alp_{mass}GeV"  # e.g. 'alp_4GeV'
    folder = os.path.join(ALP_BASE_DIR, f"{mass}GeV")
    if not os.path.isdir(folder):
        print(f"[ALP] WARNING: directory '{folder}' missing - sample skipped.")
        continue

    # Gather every *.root file inside the mass folder
    roots = sorted(f for f in os.listdir(folder) if f.endswith(".root"))
    if not roots:
        print(f"[ALP] WARNING: no ROOT files in '{folder}' - sample skipped.")
        continue

    SAMPLES[tag] = [os.path.join(f"{mass}GeV", r) for r in roots]

    # Expected events: σ·L divided by 10 000 generated MC events per file
    alp_expected[tag] = (sigma_nb * nanobarn * dataLumi) / 10000

# Merge ALP expectations into the master scaling dict
EXPECTED_COUNTS.update(alp_expected)

print(f"[ALP] Added {len(alp_expected)} ALP mass points to processing queue.")

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def extract_hist(root_file, hist_name):
    """Return (counts, bin_edges) from the key
    ``Nominal/3_LbLeventselection/Pair/<hist_name>`` or ``(None, None)`` if
    it is absent."""
    if root_file is None:
        return None, None

    key = f"{HIST_DIR_KEY_CR}/{hist_name}"
    try:
        h = root_file[key]
        counts, edges = h.to_numpy()
        return np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)
    except Exception as err:
        print(f"       Missing histogram '{key}' ({err})")
        return None, None

def extract_histCR(root_file, hist_name):
    """Return (counts, bin_edges) from the key
    ``Nominal/3_LbLeventselection/Pair/<hist_name>`` or ``(None, None)`` if
    it is absent."""
    if root_file is None:
        return None, None

    key = f"{HIST_DIR_KEY_CR}/{hist_name}"
    try:
        h = root_file[key]
        counts, edges = h.to_numpy()
        return np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)
    except Exception as err:
        print(f"       Missing histogram '{key}' ({err})")
        return None, None



def normalise_hist(counts, expected):
    """Return scaled counts so that the histogram integral equals
    ``expected`` along with the scale factor."""
    if counts is None:
        return None, 0.0  # Nothing to normalise

    integral = counts.sum()
    if integral <= 0:
        scale_factor = 0.0
        scaled_counts = counts
    else:
        scale_factor = expected / integral
        scaled_counts = counts * scale_factor

    return scaled_counts, scale_factor

# --------------------------------------------------------------------
# Path resolution helper
# --------------------------------------------------------------------

def _resolve_path(sample_tag, fname):
    """Resolve ``fname`` into an absolute path based on the sample tag."""
    if os.path.isabs(fname):
        return fname
    if sample_tag.startswith("alp_"):
        return os.path.join(ALP_BASE_DIR, fname)
    return os.path.join(BASE_DIR, fname)

# --------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------
all_histograms = defaultdict(dict)  # Top‑level dict to pickle
summary_rows = []  # For building the summary table

for sample, file_list in SAMPLES.items():
    print(f"\nProcessing sample: {sample}")

    # Aggregated histograms for all files in this sample
    sample_hists = {h: None for h in HIST_NAMES}
    sample_edges = {h: None for h in HIST_NAMES}
    sample_integrals = {h: 0.0 for h in HIST_NAMES}

    # ALP only: number of generated events over **all** ROOT files
    total_generated = 0.0

    # ---------------------------------------------------------------
    # Loop over ROOT files for the sample
    # ---------------------------------------------------------------
    for filename in file_list:
        full_path = _resolve_path(sample, filename)
        print(f"  └── Reading: {full_path}")

        root_file = None
        if uproot is not None and os.path.exists(full_path):
            try:
                root_file = uproot.open(full_path)
            except Exception as exc:
                print(f"       Could not open ROOT file ({exc}). Using dummy data.")

        for hist_name in HIST_NAMES:
            counts, edges = extract_hist(root_file, hist_name)

            # Fallback to dummy arrays if extraction failed
            if counts is None:
                counts = np.asarray([0.0], dtype=float)
                edges = np.asarray([0.0, 1.0], dtype=float)

            # Initialise storage if first encounter
            if sample_hists[hist_name] is None:
                sample_hists[hist_name] = counts
                sample_edges[hist_name] = edges
            else:
                # Sanity‑check binning consistency
                if len(edges) != len(sample_edges[hist_name]) or not np.all(edges == sample_edges[hist_name]):
                    print(f"       Inconsistent binning for '{hist_name}' - skipping.")
                    continue
                # Accumulate counts
                sample_hists[hist_name] += counts

            sample_integrals[hist_name] = sample_hists[hist_name].sum()
            
        # ── accumulate generator-level statistics (ALP + lbyl) ──────────
        if (sample.startswith("alp_") or sample == "lbyl") and root_file is not None:
            try:
                total_generated += get_total_events(root_file)
            except RuntimeError as e:
                print(f"       {e}")

    # ---------------------------------------------------------------
    # Apply normalisation (MC samples only)
    # ---------------------------------------------------------------
    if sample in EXPECTED_COUNTS:
        expected_total = EXPECTED_COUNTS[sample]

        # Use the first non‑empty histogram as reference integral
        reference_integral = next((sample_integrals[h] for h in HIST_NAMES if sample_integrals[h] > 0), None)

        if reference_integral is None:
            scale_factor = 0.0
            print("       No non-zero histograms found; scale factor set to 0.")
        else:
            scale_factor = expected_total / reference_integral

        # Apply the global scale factor to every histogram
        for h in HIST_NAMES:
            sample_hists[h] *= scale_factor
            sample_integrals[h] = sample_hists[h].sum()
    else:
        scale_factor = 1.0  # Data sample – no scaling
        
    # ---------------------------------------------------------------
    # Compute efficiency before scaling (ALP samples + lbyl)
    # ---------------------------------------------------------------
    eff_sr = None
    if (sample.startswith("alp_") or sample == "lbyl") and total_generated > 0.0:
        # first non-empty histogram integral ≡ N_pass_SR before scaling
        n_pass = next(
            (sample_integrals[h] / scale_factor for h in HIST_NAMES if sample_integrals[h] > 0),
            0.0,
        )
        eff_sr = n_pass / total_generated

    # ---------------------------------------------------------------
    # Collect information for the summary table
    # ---------------------------------------------------------------
    summary_rows.append({
        "Sample": sample,
        "Files": ", ".join(file_list),
        "Scale factor": round(scale_factor, 4),
        "Reference integral": round(sample_integrals[HIST_NAMES[0]], 2),
        "Expected events": EXPECTED_COUNTS.get(sample, "n/a"),
        "Sig eff": eff_sr,
    })

    # ---------------------------------------------------------------
    # Store histograms into the big dictionary
    # ---------------------------------------------------------------
    all_histograms[sample] = {
        hist_name: {
            "bin_edges": sample_edges[hist_name].tolist(),
            "counts": sample_hists[hist_name].tolist(),
        }
        for hist_name in HIST_NAMES
    }

    # One extra key per ALP mass point
    if eff_sr is not None:
        all_histograms[sample]["eff_sr"] = eff_sr

# --------------------------------------------------------------------
# Pickle the final histogram dictionary
# --------------------------------------------------------------------
output_dir = "bkg"
os.makedirs(output_dir, exist_ok=True)

# ---------- signal-region pickle (renamed) ----------
pickle_path = os.path.join(output_dir, "bkg_alp_sr_pickle.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(all_histograms, f)

print(f"\nSignal-region histograms saved to '{pickle_path}'.")

# --------------------------------------------------------------------
# === SECOND PASS : control-region samples ===========================
# --------------------------------------------------------------------
print("\nSummary:")
cr_histograms   = defaultdict(dict)
cr_summary_rows = []

for sample, file_list in CR_SAMPLES.items():
    print(f"\nProcessing CR sample: {sample}")

    sample_hists      = {h: None for h in HIST_NAMES}
    sample_edges      = {h: None for h in HIST_NAMES}
    sample_integrals  = {h: 0.0  for h in HIST_NAMES}

    for filename in file_list:
        full_path = os.path.join(CR_BASE_DIR, filename)
        print(f"  └── Reading: {full_path}")

        root_file = None
        if uproot is not None and os.path.exists(full_path):
            try:
                root_file = uproot.open(full_path)
            except Exception as exc:
                print(f"       Could not open ROOT file ({exc}). Using dummy data.")

        for hist_name in HIST_NAMES:
            counts, edges = extract_histCR(root_file, hist_name)
            if counts is None:                       # fallback dummy
                counts = np.asarray([0.0])
                edges  = np.asarray([0.0, 1.0])

            if sample_hists[hist_name] is None:      # first file
                sample_hists[hist_name] = counts
                sample_edges[hist_name] = edges
            else:
                if len(edges) != len(sample_edges[hist_name]) or not np.all(edges == sample_edges[hist_name]):
                    print(f"       Inconsistent binning for '{hist_name}' - skipping.")
                    continue
                sample_hists[hist_name] += counts

            sample_integrals[hist_name] = sample_hists[hist_name].sum()

    # -- control-region normalisation --------------------------------
    if sample in EXPECTED_CR_COUNTS:
        expected_total     = EXPECTED_CR_COUNTS[sample]
        reference_integral = next(
            (sample_integrals[h] for h in HIST_NAMES if sample_integrals[h] > 0), None
        )
        if reference_integral is None:
            scale_factor = 0.0
            print("       No non-zero histograms found; scale factor set to 0.")
        else:
            scale_factor = expected_total / reference_integral

        for h in HIST_NAMES:
            sample_hists[h] *= scale_factor
            sample_integrals[h] = sample_hists[h].sum()
    else:
        scale_factor = 1.0       # data

    cr_summary_rows.append({
        "Sample":            sample,
        "Files":             ", ".join(file_list),
        "Scale factor":      round(scale_factor, 4),
        "Reference integral":round(sample_integrals[HIST_NAMES[0]], 2),
        "Expected events":   EXPECTED_CR_COUNTS.get(sample, "n/a"),
    })

    cr_histograms[sample] = {
        hist_name: {
            "bin_edges": sample_edges[hist_name].tolist(),
            "counts":    sample_hists[hist_name].tolist(),
        }
        for hist_name in HIST_NAMES
    }



# ---------- control-region pickle ----------
cr_pickle_path = os.path.join(output_dir, "bkg_cr_pickle.pkl")
with open(cr_pickle_path, "wb") as f:
    pickle.dump(cr_histograms, f)

print(f"\nControl-region histograms saved to '{cr_pickle_path}'.")

# --------------------------------------------------------------------
# (optional) print summaries
# --------------------------------------------------------------------
print("\nSR summary:")
for row in summary_rows:
    print(row)

print("\nCR summary:")
for row in cr_summary_rows:
    print(row)