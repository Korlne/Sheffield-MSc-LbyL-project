#!/usr/bin/env python
import os
import re
import pickle
import argparse
import multiprocessing
import functools
import ROOT
from lbyl_common import get_sample_file_list, process_events, print_cutflow, print_aco_scan
# =============================================================================
#  Batch-driver for the LbyL analysis                                
#  ============================================================================ 
#
#  What this driver does
#  ---------------------
#  • Walks through the **AOD file list** delivered by *lbyl_common.get_sample_file_list()*  
#    (signal, CEP, γγ→e⁺e⁻, ALP, …) and farms out each file to *process_events*  
#    — the heavy-lifting event-loop defined in **lbyl_common.py**.
#  • Uses the Python *multiprocessing* pool to process many AODs in parallel
#    (one worker per logical CPU core by default).
#  • Saves the **cut-flow dictionary and per-event arrays** as a
#    **Pickle** for later plotting / statistics.
#  • Supports two work-flows  
#        1. **Merged** mode *(default)* – all files combined into one pickle  
#        2. **One-pickle-per-file** mode – currently activated for bulk ALP
#           samples (`--sample alp`) and for future graviton samples.
#
#  Pickle output – what’s inside?
#  ------------------------------
#  The merged pickle (`cutflow_<sample>_aco-<sr|cr|sr_scan>.pkl`) is exactly
#  the *dictionary* returned by **merge_results**:
#
#      {
#          "cut_flow"               : {step → int},
#          "diphoton_masses"        : [...],
#          "diphoton_pts"           : [...],
#          "diphoton_acoplanarity"  : [...],
#          "leading_photon_ets"     : [...],
#          "diphoton_rapidity_diff" : [...],
#          "diphoton_cos_thetas"    : [...],
#          "event_weights"          : [...],
#          "aco_scan"               : {"aco<=0.005": N, …},  # present only in sr_scan mode
#      }
#
#  When the *one-pickle-per-file* scheme is triggered (ALP bulk mode) each file
#  is stored separately as `cutflow_alp<MASS>_aco-<mode>.pkl`, containing the
#  identical dictionary produced by *process_events*.
#
#  Command-line interface
#  ----------------------
#  ```bash
#  # Signal region (A_φ ≤ 0.01) for the inclusive “signal” MC
#  $ python process_cutflow.py --sample signal --aco sr
#
#  # Control region (A_φ ≥ 0.01)
#  $ python process_cutflow.py --sample cep --aco cr
#
#  # Full acoplanarity-threshold scan on data-driven ALP-mass grid
#  $ python process_cutflow.py --sample alp --aco sr_scan
#
#  # Work on one specific ALP mass point only (here: 6 GeV)
#  $ python process_cutflow.py --sample alp6 --aco sr
#  ```
#
#  CLI options
#  -----------
#    --sample   ‘signal’, ‘yyee’, ‘cep’, ‘alp’, **or** a single ALP mass tag  
#    --aco      ‘sr’ (signal region), ‘cr’ (control region), ‘sr_scan’ (fill
#               running optimisation counters)
#
#  Implementation notes
#  --------------------
#  * Each worker initialises **xAODRootAccess** locally (`ROOT.xAOD.Init()`),
#    builds a transient tree, and calls *lbyl_common.process_events*.
#  * ALP helper `_alp_label_from_file()` extracts the mass either from the file
#    name (`ma5`) or, failing that, from the dataset-ID lookup table.
#  * The graviton helper is stubbed out and will raise if called – extend once
#    samples exist.
#  * The final pickle is written in the current working directory; cut-flow
#    tables are echoed to the console via *lbyl_common.print_cutflow*.
#
#  Dependencies
#  ------------
# ATLAS Athena AnalysisBase,25.2.15
# AlmaLinux 9
#  
#
#  Last updated : 24 Jun 2025
# =============================================================================



# -----------------------------------------------------------------------------
# ALP and gtaviton 
# -----------------------------------------------------------------------------
_ALP_MASS_RE = re.compile(r"ma(\d+)")
_ALP_TID2MASS = {
    "41382322": "4",
    "41382338": "5",
    "41382354": "6",
    "41382371": "7",
    "41382387": "8",
    "41382404": "9",
    "41382420": "10",
    "41382437": "12",
    "41382453": "14",
    "41382469": "15",
    "41382486": "16",
    "41382504": "18",
    "41382520": "20",
    "41382554": "40",
    "41382572": "50",
    "41382589": "60",
    "41382606": "70",
    "41382623": "80",
    "41382640": "90",
    "41382657": "100",
    # extend for new masses
}


def _alp_label_from_file(path: str) -> str:
    """Return 'alp<MASS>' derived from a *.pool.root.1 file name."""
    fname = os.path.basename(path)
    m = _ALP_MASS_RE.search(fname)
    if m:
        return f"alp{m.group(1)}"
    tid = fname.split(".")[1]
    return f"alp{_ALP_TID2MASS.get(tid, tid)}"

def _graviton_label_from_file(path: str) -> str: 
    """
    Fill graviton
    """
    raise NotImplementedError("Implement when graviton samples are available")

def process_file(file_path, aco_mode='cr'):
    # Initialize xAOD in the worker process.
    ROOT.xAOD.Init().ignore()
    
    # Create a TChain with the single file.
    chain = ROOT.TChain("CollectionTree")
    chain.Add(file_path)
    import xAODRootAccess.GenerateDVIterators
    t = ROOT.xAOD.MakeTransientTree(chain)
    
    
    
    # Process events in this file.
    result = process_events(t, aco_mode=aco_mode)
    return result

# -----------------------------------------------------------------------------
# one-pickle-per-file
# -----------------------------------------------------------------------------
def process_independent_files(
    file_list,
    label_from_file,
    aco_mode="sr",
):
    """
    Iterate over *file_list*, run the selection. Designed for alp & graviton samples (e.g. --sample 'alp5').
    """
    for f in file_list:
        sub_sample = label_from_file(f)

        # 1) run selection -----------------------------------------------------
        result = process_file(f, aco_mode=aco_mode)

        # 2) cut-flow pickle ---------------------------------------------------
        cutflow_pickle = f"cutflow_{sub_sample}_aco-{aco_mode}.pkl"
        with open(cutflow_pickle, "wb") as pf:
            pickle.dump(result, pf)
        print(f"Cut-flow for {sub_sample} ({aco_mode}) saved to {cutflow_pickle}")

        # 3) console summary ---------------------------------------------------
        print_cutflow(sub_sample, result["cut_flow"])
        if aco_mode == "sr_scan":
            print_aco_scan(result["aco_scan"], header="Acoplanarity scan")

# Dictionary for merged result
def merge_results(results):
    merged = {
        "cut_flow": {},
        "diphoton_masses": [],
        "diphoton_pts": [],
        "diphoton_acoplanarity": [],
        "leading_photon_ets": [],
        "diphoton_rapidity_diff": [],
        "diphoton_cos_thetas": [],
        "event_weights": [],
        "aco_scan": {}              # aco dictionary
    }
    for r in results:
        for key, value in r["cut_flow"].items():
            merged["cut_flow"][key] = merged["cut_flow"].get(key, 0) + value
        merged["diphoton_masses"].extend(r["diphoton_masses"])
        merged["diphoton_pts"].extend(r["diphoton_pts"])
        merged["diphoton_acoplanarity"].extend(r["diphoton_acoplanarity"])
        merged["leading_photon_ets"].extend(r.get("leading_photon_ets", []))
        merged["diphoton_rapidity_diff"].extend(r.get("diphoton_rapidity_diff", []))
        merged["diphoton_cos_thetas"].extend(r.get("diphoton_cos_thetas", []))
        merged["event_weights"].extend(r.get("event_weights", []))
        for k, v in r.get("aco_scan", {}).items():
            merged["aco_scan"][k] = merged["aco_scan"].get(k, 0) + v
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Generate tree info and process cutflow for a specific sample using multiprocessing."
    )
    parser.add_argument(
        "--sample",
        default="signal",
        help=(
            "Sample to process: signal | yyee | cep | alp  (all ALP masses) | "
            "alpx (single mass, e.g. alp5, alp12 for run individual alp mass)"
        ),
    )
    parser.add_argument("--aco", choices=[ "sr", "cr", "sr_scan"],
                        default="sr", help="Acoplanarity in signal region aco<=0.01 or controlled region aco>=0.01")
    args = parser.parse_args()
    
    sample = args.sample.lower()
    aco_mode = args.aco.lower()
    
    file_list = get_sample_file_list(sample)
    if not file_list:
        print("No files found for sample", sample)
        return
    
    # ------------------------------------------------------------------
    # 1) ALP bulk mode → independent-file workflow
    # ------------------------------------------------------------------
    if sample == "alp":
        process_independent_files(file_list, _alp_label_from_file, aco_mode)
        return

    # ------------------------------------------------------------------
    # 2) Future graviton bulk mode (stub)
    # ------------------------------------------------------------------
    if sample == "graviton":  # pragma: no cover
        process_independent_files(file_list, _graviton_label_from_file, aco_mode)
        return
    
    # Process files in parallel if there are multiple files, otherwise process the single file.
    if len(file_list) > 1:
        num_processes = multiprocessing.cpu_count()
        worker = functools.partial(process_file, aco_mode=aco_mode)
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(worker, file_list)
    else:
        results = [process_file(file_list[0], aco_mode=aco_mode)]
    
    merged_results = merge_results(results)
    
    # ----------------------------------------------
    # Save merged cutflow results.
    cutflow_pickle = f"cutflow_{sample}_aco-{aco_mode}.pkl"
    with open(cutflow_pickle, "wb") as f:
        pickle.dump(merged_results, f)
    print(f"Cut-flow for sample {sample} ({aco_mode}) saved to {cutflow_pickle}")

    # Print summary table ----------------------------------------------
    print_cutflow(sample, merged_results["cut_flow"])
    if aco_mode == "sr_scan":                # only meaningful in scan mode
        print_aco_scan(merged_results["aco_scan"],
                    header="Acoplanarity scan")

if __name__ == "__main__":
    main()
