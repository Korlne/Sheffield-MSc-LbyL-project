#!/usr/bin/env python
"""Refactored LbyL analysis:
• Kinematic arrays store *raw* (un‑weighted) values.
• Per‑event weights are collected separately in `event_weights` and can be
  passed as the `weights=` argument when histogramming so only the bin
  *height* is scaled.
"""
import os
import ROOT
import ctypes
import math
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Conversion factor
GeV = 1000.0

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def invariant_mass(obj1, obj2):
    """Invariant mass of two xAOD objects assumed massless (photons)."""
    p1, p2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
    p1.SetPtEtaPhiM(obj1.pt(), obj1.eta(), obj1.phi(), 0)
    p2.SetPtEtaPhiM(obj2.pt(), obj2.eta(), obj2.phi(), 0)
    return (p1 + p2).M()


# ----------------------------------------------------------------------
# Functions ported from EXCLRunII C++ (unchanged)
# ----------------------------------------------------------------------

def passes_bl_spacepoint_preselection(space_point, photons):
    for photon in photons:
        dphi = ROOT.TVector2.Phi_mpi_pi(photon.phi() - space_point.Phi())
        if abs(dphi) > 1.5:
            continue
        if abs(photon.eta() - space_point.Eta()) > 0.05:
            continue
        return True
    return False


def l1_trigger_weight(cluster1, cluster2, params):
    if cluster1 is None or cluster2 is None:
        return 1.0
    sum_et = (cluster1.pt() + cluster2.pt()) / GeV
    p0, p1, p2 = params
    arg = (sum_et - p0) / (p1 + p2 * sum_et)
    return 0.5 * (ROOT.TMath.Erf(arg) + 1.0)


def hlt_weight(track1, track2, params):
    if track1 is None or track2 is None:
        return 1.0
    vec1, vec2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
    vec1.SetPtEtaPhiM(track1.pt(), track1.eta(), track1.phi(), track1.m())
    vec2.SetPtEtaPhiM(track2.pt(), track2.eta(), track2.phi(), track2.m())
    y = (vec1 + vec2).Rapidity()
    return params[0] * y * y + params[1] * y + params[2]


def scale_factor_weight(p1, p2, hist):
    if not hist:
        return 1.0
    xbins = hist.GetXaxis().GetNbins()
    b1 = hist.FindBin(p1.pt() / GeV, p1.eta())
    b2 = hist.FindBin(p2.pt() / GeV, p2.eta())
    sf1 = hist.GetBinContent(b1 % xbins, b1 // xbins)
    sf2 = hist.GetBinContent(b2 % xbins, b2 // xbins)
    return 1.0 if sf1 == 0 or sf2 == 0 else sf1 * sf2


# ----------------------------------------------------------------------
# TMVA reader cache for photon ID (unchanged)
# ----------------------------------------------------------------------
_nn_readers = {}

def get_nn_reader(category):
    if category in _nn_readers:
        return _nn_readers[category]

    reader = ROOT.TMVA.Reader("!Color:Silent")
    # Allocate c_float slots
    f1, weta2, eratio, fracs1, f1core = (ctypes.c_float() for _ in range(5))
    reader.AddVariable("photon_f1", ctypes.byref(f1))
    reader.AddVariable("photon_weta2", ctypes.byref(weta2))
    reader.AddVariable("photon_Eratio", ctypes.byref(eratio))
    reader.AddVariable("photon_fracs1", ctypes.byref(fracs1))
    reader.AddVariable("photon_f1core", ctypes.byref(f1core))

    weights = f"DNN_weights/weights_DNN_PbPb_data_train_9k_eta{category}.xml"
    reader.BookMVA("Signal", weights)

    _nn_readers[category] = (reader, (f1, weta2, eratio, fracs1, f1core))
    return _nn_readers[category]


# ----------------------------------------------------------------------
# Sample‑file helper (unchanged)
# ----------------------------------------------------------------------

def get_sample_file_list(sample):
    s = sample.lower()
    # ... unchanged implementation ...


# ----------------------------------------------------------------------
# Main event loop
# ----------------------------------------------------------------------

def process_events(tree, aco_mode="cr"):
    """Loop over events, filling cut‑flow and kinematic arrays.

    Returns a dict with raw kinematics and a matching list `event_weights` so
    downstream code can feed, e.g. `hist(..., weights=event_weights)`.
    """

    # Initialise external tools (TrigDecisionTool, etc.) – implementation
    # remains the same as the original script; omitted here for brevity.

    # -----------------------------
    # Containers / bookkeeping
    # -----------------------------
    cut_flow = {
        "Total Events": 0,
        "Pass Track Veto": 0,
        "Pass Pixel Track Veto": 0,
        "Pass Photon eta cut": 0,
        "Pass Photon pt cut": 0,
        "Pass Photon Selection": 0,
        "Pass Diphoton Selection": 0,
        "Pass Acoplanarity Selection": 0,
        "Pass Electron Selection": 0,
    }

    # Per‑event kinematics (UN‑WEIGHTED)
    diphoton_masses, diphoton_pts, diphoton_acoplanarity = [], [], []
    leading_photon_ets = []
    diphoton_rapidity_diff, diphoton_cos_thetas = [], []

    # Per‑event weight (one‑to‑one with arrays above)
    event_weights = []

    # Photon‑level outputs (un‑weighted)
    photon_pt, photon_eta = [], []
    photon_idNN_out, photon_idNN_pass = [], []

    # -----------------------------
    # Event loop (details unchanged)
    # -----------------------------
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        cut_flow["Total Events"] += 1

        # ... all selection, calibration, NN ID logic unchanged ...

        # After final selection -------------------------------
        # Compute event weight
        total_weight = l1_trigger_weight(cluster1, cluster2, (4.5, 1.0, 0.5))
        total_weight *= hlt_wt
        total_weight *= scale_factor_weight(lead, sub, sf_hist)

        # Build TLorentzVectors for final kinematics
        p1, p2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
        p1.SetPtEtaPhiM(lead.pt(), lead.eta(), lead.phi(), 0)
        p2.SetPtEtaPhiM(sub.pt(),  sub.eta(),  sub.phi(),  0)

        # ---------------- Raw values ----------------
        m_gg        = invariant_mass(lead, sub) / GeV
        pt_sum      = (p1 + p2).Pt() / GeV
        dphi        = abs(lead.phi() - sub.phi()); dphi = dphi if dphi <= math.pi else 2*math.pi - dphi
        aco_phi     = 1 - dphi / math.pi
        e_lead      = p1.E(); et_lead = e_lead / math.cosh(p1.Eta()) / GeV
        dy          = p1.Rapidity() - p2.Rapidity()
        cos_theta   = abs(math.tanh(dy/2))

        # ---------------- Store ----------------------
        diphoton_masses.append(m_gg)
        diphoton_pts.append(pt_sum)
        diphoton_acoplanarity.append(aco_phi)
        leading_photon_ets.append(et_lead)
        diphoton_rapidity_diff.append(dy)
        diphoton_cos_thetas.append(cos_theta)

        event_weights.append(total_weight)

    # -----------------------------
    # Return
    # -----------------------------
    return {
        "cut_flow": cut_flow,
        "diphoton_masses":        diphoton_masses,
        "diphoton_pts":           diphoton_pts,
        "diphoton_acoplanarity":  diphoton_acoplanarity,
        "leading_photon_ets":     leading_photon_ets,
        "diphoton_rapidity_diff": diphoton_rapidity_diff,
        "diphoton_cos_thetas":    diphoton_cos_thetas,
        "event_weights":          event_weights,
        # Photon‑level
        "photon_pt":        photon_pt,
        "photon_eta":       photon_eta,
        "photon_idNN_out":  photon_idNN_out,
        "photon_idNN_pass": photon_idNN_pass,
    }


# ----------------------------------------------------------------------
# Cut‑flow printer (unchanged except for docstring)
# ----------------------------------------------------------------------

def print_cutflow(sample_label, cut_flow):
    print(f"\nCutflow for {sample_label}:")
    total = cut_flow.get("Total Events", 0)
    order = [
        ("All events",           "Total Events"),
        ("Photon eta cut",       "Pass Photon eta cut"),
        ("Photon pt cut",        "Pass Photon pt cut"),
        ("Photon selection",     "Pass Photon Selection"),
        ("Diphoton selection",   "Pass Diphoton Selection"),
        ("Track veto",           "Pass Track Veto"),
        ("Pixel veto",           "Pass Pixel Track Veto"),
        ("Electron sel.",        "Pass Electron Selection"),
        ("Acoplanarity sel.",    "Pass Acoplanarity Selection"),
    ]
    for label, key in order:
        passed = cut_flow.get(key, 0)
        if key == "Pass Electron Selection":
            print(f"{label:25s} {passed:7d}")
        else:
            eff = (passed / total * 100) if total else 0
            err = (100 * math.sqrt(eff/100 * (1 - eff/100) / total)) if total else 0
            print(f"{label:25s} {passed:7d}  {eff:6.2f}% ± {err:.2f}%")
