#!/usr/bin/env python3
"""
get_aco_scan_result.py  (auto-discover ALP samples)

Compute the significance curve

    Z(thr) = S / sqrt(B)

versus the diphoton-acoplanarity cut where

  • S  = signal yield from each ALP sample (cutflow_*_aco-sr_scan.pkl)
  • B  = LbyL + CEP + yy→ee backgrounds

------------------------------------------------------------------
Key features
------------------------------------------------------------------
* **Automatic discovery**: if no explicit `--alp` files are given, the
  script searches a directory (default: the current working directory)
  for files matching the glob pattern ``cutflow_*_aco-sr_scan.pkl``.
* **Per-sample plots**: a stepped-line plot of the significance curve is
  produced **for each ALP sample individually**.  The default filenames
  are ``<sample>_aco_significance.png`` but can be customised via
  `--plot`.
* **Tabular summary**: for every sample the script prints a table of
  S, B and Z for each acoplanarity threshold and highlights the
  threshold with maximum significance.
"""

import argparse
import re
import math
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Sequence
from AthenaCommon.SystemOfUnits import nanobarn
import mplhep as hep                
hep.style.use(hep.style.ATLASAlt)

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Default background pickle locations
# ---------------------------------------------------------------------------
DEFAULT_LBYL = Path("/home/jtong/lbyl/cutflow_signal_aco-sr_scan.pkl")
DEFAULT_CEP  = Path("/home/jtong/lbyl/cutflow_cep_aco-sr_scan.pkl")
DEFAULT_YYEE = Path("/home/jtong/lbyl/yyee_binned/yyee_root_merged_hist_aco-sr.pkl")
DEFAULT_PLOT = "aco_significance.png"          # used as a suffix base
DEFAULT_GLOB = "cutflow_alp*_aco-sr_scan.pkl"    # pattern for autodiscovery

alp_sigma_nb = {
    4: 7.967330e3, 5: 6.953744e3, 6: 6.044791e3, 7: 5.300250e3,
    8: 4.670220e3, 9: 4.154600e3, 10: 3.709976e3, 12: 3.016039e3,
    14: 2.499097e3, 15: 2.285133e3, 16: 2.093761e3, 18: 1.782345e3,
    20: 1.526278e3, 30: 7.779030e2, 40: 4.368360e2, 50: 2.600118e2,
    60: 1.604056e2, 70: 1.016849e2, 80: 6.546058e1, 90: 4.280824e1,
    100: 2.824225e1,
}

cross_section_signal = 879.2621 * nanobarn


# Integrated luminosity (≈ 1.67 nb⁻¹)
dataLumi = 1.67 / nanobarn


# ---------------------------------------------------------------------------
#  Command‑line interface
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute S/√B vs acoplanarity requirement for ALP signal samples. "
            "If no ALP pickles are given explicitly, the script scans a "
            "directory for files matching the default pattern.  A stepped "
            "significance plot is created *per* sample."
        )
    )

    alp_group = p.add_mutually_exclusive_group(required=False)
    alp_group.add_argument(
        "--alp", nargs="+", metavar="PICKLE",
        help="one or more ALP signal pickles (cutflow_*_aco-sr_scan.pkl)"
    )
    alp_group.add_argument(
        "--scan-dir", metavar="DIR", default=".",
        help="directory in which to search for ALP pickles (default: current directory)"
    )

    p.add_argument("--glob", default=DEFAULT_GLOB,
                   help=f"glob pattern for autodiscovery (default: '{DEFAULT_GLOB}')")

    p.add_argument("--lbyl", default=str(DEFAULT_LBYL),
                   help="override LbyL background pickle")
    p.add_argument("--cep",  default=str(DEFAULT_CEP),
                   help="override CEP background pickle")
    p.add_argument("--yyee", default=str(DEFAULT_YYEE),
                   help="override yyee background pickle")

    p.add_argument(
        "--plot", metavar="PNG", default=DEFAULT_PLOT,
        help=(
            "base filename for significance plots (default: %(default)s). "
            "For multiple samples, the sample name is inserted before the "
            "file extension."
        ),
    )
    p.add_argument("--no-plot", action="store_true",
                   help="suppress plot generation altogether")
    return p.parse_args()

# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def threshold_to_float(key: str) -> float:
    """Convert keys like '0p005' or 'aco<=0.005' to a float (0.005)."""
    if "aco<=" in key:
        key = key.split("<=")[-1]
    return float(key.replace("p", "."))


def format_slice_key(thr: float) -> str:
    """Convert 0.005 ➜ '0p005' (as used in the yy→ee pickle)."""
    return f"{thr:.3f}".replace(".", "p")


def yield_from_scan_dict(scan_dict: dict, thr: float) -> float:
    key_full = f"aco<={thr:.3f}"
    if key_full in scan_dict:
        return scan_dict[key_full]
    key_trim = key_full.rstrip("0").rstrip(".")
    return scan_dict.get(key_trim, 0.0)


def yield_from_arrays(sample: dict, thr: float) -> float:
    aco = sample["diphoton_acoplanarity"]
    w   = sample.get("event_weights")
    if w is not None:
        return sum(wx for ax, wx in zip(aco, w) if ax <= thr)
    return sum(1 for ax in aco if ax <= thr)


def get_yield_generic(path: Path, thr: float) -> float:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "aco_scan" in data:
        return yield_from_scan_dict(data["aco_scan"], thr)
    return yield_from_arrays(data, thr)


def get_yield_yyee(path: Path, thr: float) -> float:
    with open(path, "rb") as f:
        yyee = pickle.load(f)
    key = format_slice_key(thr)
    if key not in yyee:
        return 0.0
    hist = yyee[key]["h_ZMassZoom"]
    return sum(hist["counts"])

# ---------------------------------------------------------------------------
#  Significance calculation for a single ALP sample
# ---------------------------------------------------------------------------

def compute_significance(alp_path: Path, thresholds: Sequence[float], args) -> Dict[float, Dict[str, float]]:
    """Return OrderedDict[thr] -> {'S':…, 'B':…, 'Z':…}."""
    res = OrderedDict()
    for thr in thresholds:
        s = get_yield_generic(alp_path, thr)
        b_tot = (
            get_yield_generic(args.lbyl, thr)
            + get_yield_generic(args.cep, thr)
            + get_yield_yyee(args.yyee, thr)
        )
        z = s / math.sqrt(b_tot) if b_tot > 0 else float("inf")
        res[thr] = dict(S=s, B=b_tot, Z=z)
    return res

# ---------------------------------------------------------------------------
#  Utility: derive clean sample name from filename
# ---------------------------------------------------------------------------

def sample_name_from_path(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    # try to return the part that looks like 'alpX' if present
    for part in parts:
        if part.lower().startswith("alp"):
            return part
    return stem

# ---------------------------------------------------------------------------
#  Plotting helper
# ---------------------------------------------------------------------------

def save_significance_plot(xs: List[float], zs: List[float], sample_name: str, outfile: Path):
    plt.figure(figsize=(6, 4))
    plt.step(xs, zs, where="post")
    plt.xlabel("Diphoton acoplanarity cut (aco ≤ thr)")
    plt.ylabel("Significance S/√B")
    plt.title(f"Significance vs acoplanarity cut – {sample_name}")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"Plot saved to {outfile}")

# ---------------------------------------------------------------------------
#  Main routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---------------------------------------------------------------------
    #  Determine list of ALP sample files
    # ---------------------------------------------------------------------
    if args.alp:
        alp_files = [Path(p) for p in args.alp]
    else:
        search_dir = Path(args.scan_dir).expanduser().resolve()
        alp_files = sorted(search_dir.glob(args.glob))
        if not alp_files:
            raise SystemExit(f"No ALP pickles matching '{args.glob}' found in {search_dir}")

    # ---------------------------------------------------------------------
    #  Build common threshold grid from yy→ee pickle
    # ---------------------------------------------------------------------
    with open(args.yyee, "rb") as f:
        yyee_data = pickle.load(f)
    thresholds = sorted(threshold_to_float(k) for k in yyee_data)
    # Container to hold (mass, best_thr) for each sample
    mass_thr_pairs = []

    # ---------------------------------------------------------------------
    #  Process each ALP sample
    # ---------------------------------------------------------------------
    hdr_fmt = "{:<21} {:>11} {:>12} {:>12}"
    row_fmt = "{:<21} {:>11.2f} {:>12.2f} {:>12.3f}"

    base_plot = Path(args.plot)

    for alp_path in alp_files:
        sample_name = sample_name_from_path(alp_path)
        results = compute_significance(alp_path, thresholds, args)
        best_thr, best = max(results.items(), key=lambda kv: kv[1]["Z"])
        
        # Pull the mass out of the sample name “alpXX…”
        m = re.search(r'\d+', sample_name)
        if m:
            mass_thr_pairs.append((float(m.group()), best_thr))
            
        # -------------------- print table --------------------
        print("\n" + "=" * 80)
        print(f"Signal sample: {sample_name}")
        print("Backgrounds   : LbyL, CEP, yy→ee (slice 0p010)")
        print(hdr_fmt.format("Aco bin", "S", "B", "S/√B"))
        print("-" * 54)
        for thr, r in results.items():
            print(row_fmt.format(f"aco<={thr:.3f}", r["S"], r["B"], r["Z"]))
        print("-" * 54)
        print(
            f"Highest S/√B: {best['Z']:.3f} at aco<={best_thr:.3f}  "
            f"(S = {best['S']:.2f}, B = {best['B']:.2f})\n"
        )

        # -------------------- plot curve --------------------
        if not args.no_plot:
            xs = list(results.keys())
            zs = [r["Z"] for r in results.values()]
            if len(alp_files) == 1:
                outfile = base_plot
            else:
                outfile = base_plot.with_name(f"{sample_name}_{base_plot.name}")
            save_significance_plot(xs, zs, sample_name, outfile)
            
    #  plot best_thr vs mass
    if mass_thr_pairs:
        mass_thr_pairs.sort()
        masses, best_thrs = zip(*mass_thr_pairs)

        plt.figure(figsize=(8, 6))         
        plt.plot(masses, best_thrs,
                    marker='o',
                    linestyle='-',
                    color='tab:blue',
                    label=r"$m_a$ scan")
        plt.title('Optimal Aco cut vs ALP mass')
        plt.xlabel("ALP mass (GeV)", labelpad=10)
        plt.ylabel("Optimal acoplanarity thr")
        plt.grid(True, linestyle=":")
        plt.margins(x=0.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig("aco_opt_vs_mass.png")
        plt.close()
        print("Plot saved to aco_opt_vs_mass.png")

if __name__ == "__main__":
    main()
