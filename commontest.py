#!/usr/bin/env python
import os, pickle
import ROOT, ctypes, math, numpy as np, matplotlib.pyplot as plt, mplhep as hep
#TrigDecisionTool, egtool, EGCalibrationAndSmearingTool = None, None, None

# Conversion factor
GeV = 1000.0

def invariant_mass(obj1, obj2):
    p1 = ROOT.TLorentzVector()
    p2 = ROOT.TLorentzVector()
    # Assuming massless particles (photons)
    p1.SetPtEtaPhiM(obj1.pt(), obj1.eta(), obj1.phi(), 0)
    p2.SetPtEtaPhiM(obj2.pt(), obj2.eta(), obj2.phi(), 0)
    return (p1 + p2).M()


# ---------- New Functions from EXCLRunII C++ ----------

def passes_bl_spacepoint_preselection(space_point, photons):
    """Return True if any photon is close to the space point in φ and η."""
    for photon in photons:
        dphi = ROOT.TVector2.Phi_mpi_pi(photon.phi() - space_point.Phi())
        if abs(dphi) > 1.5:
            continue
        if abs(photon.eta() - space_point.Eta()) > 0.05:
            continue
        return True
    return False

def l1_trigger_weight(cluster1, cluster2, parameters):
    """Compute L1 trigger weight from two clusters and tuple parameters."""
    if cluster1 is None or cluster2 is None:
        return 1.0
    sum_et = (cluster1.pt() + cluster2.pt()) / GeV
    p0, p1, p2 = parameters
    arg = (sum_et - p0) / (p1 + p2 * sum_et)
    return 0.5 * (ROOT.TMath.Erf(arg) + 1.0)

def hlt_weight(track1, track2, parameters):
    """Compute HLT weight from two tracks and tuple parameters."""
    if track1 is None or track2 is None:
        return 1.0
    vec1 = ROOT.TLorentzVector()
    vec2 = ROOT.TLorentzVector()
    vec1.SetPtEtaPhiM(track1.pt(), track1.eta(), track1.phi(), track1.m())
    vec2.SetPtEtaPhiM(track2.pt(), track2.eta(), track2.phi(), track2.m())
    system = vec1 + vec2
    y = system.Rapidity()
    return parameters[0]*y*y + parameters[1]*y + parameters[2]

def scale_factor_weight(particle1, particle2, scale_hist):
    """Look up scale factors for two particles from a 2D histogram."""
    if not scale_hist:
        return 1.0
    xbins = scale_hist.GetXaxis().GetNbins()
    bin1 = scale_hist.FindBin(particle1.pt() / GeV, particle1.eta())
    bin2 = scale_hist.FindBin(particle2.pt() / GeV, particle2.eta())
    sf1 = scale_hist.GetBinContent(bin1 % xbins, bin1 // xbins)
    sf2 = scale_hist.GetBinContent(bin2 % xbins, bin2 // xbins)
    if sf1 == 0.0 or sf2 == 0.0:
        return 1.0
    return sf1 * sf2

#Need to change this to path and histogram name
sf_file = ROOT.TFile.Open("scale_factorsPID_syst-2.root")
sf_hist = sf_file.Get("scale_factor_hist")

# ------------------------------------------------------

# TMVA Reader Cache for photon ID
_nn_readers = {}

def get_nn_reader(category):
    """
    Return a booked TMVA.Reader for the given photon-category.
    Readers are cached in _nn_readers so BookMVA() is only called once.
    """
    if category in _nn_readers:
        return _nn_readers[category]

    reader = ROOT.TMVA.Reader("!Color:Silent")
    # Allocate five ctypes.c_float slots for the inputs
    #f1, eratio, fracs1, e277, f1core = (ctypes.c_float() for _ in range(5))
    f1, weta2, eratio, fracs1, f1core = (ctypes.c_float() for _ in range(5))
    # reader.AddVariable("photon_f1",     ctypes.byref(f1))
    # reader.AddVariable("photon_Eratio", ctypes.byref(eratio))
    # reader.AddVariable("photon_fracs1", ctypes.byref(fracs1))
    # reader.AddVariable("photon_e277",   ctypes.byref(e277))
    # reader.AddVariable("photon_f1core", ctypes.byref(f1core))
    reader.AddVariable("photon_f1",     ctypes.byref(f1))
    reader.AddVariable("photon_weta2",  ctypes.byref(weta2))
    reader.AddVariable("photon_Eratio", ctypes.byref(eratio))
    reader.AddVariable("photon_fracs1", ctypes.byref(fracs1))
    reader.AddVariable("photon_f1core", ctypes.byref(f1core))

    # Build the path to the XML weight file for this eta-category
    weights = f"DNN_weights/weights_DNN_PbPb_data_train_9k_eta{category}.xml"
    reader.BookMVA("Signal", weights)

    #_nn_readers[category] = (reader, (f1, eratio, fracs1, e277, f1core))
    _nn_readers[category] = (reader, (f1, weta2, eratio, fracs1, f1core))
    return _nn_readers[category]

# =====================================
# Sample file-list helper
# =====================================
def get_sample_file_list(sample):
    s = sample.lower()
    if s == "signal":
        sig_dir = "/home/jtong/lbyl/data/sig/"
        return [os.path.join(sig_dir, f)
                for f in os.listdir(sig_dir)
                if f.startswith("AOD.40981863.") and f.endswith(".pool.root.1")]
    elif s == "yyee":
        yyee_dir = "/home/jtong/lbyl/data/yyee/"
        return [os.path.join(yyee_dir, "AOD.41611872._000001.pool.root.1")]
    elif s == "cep":
        cep_dir = "/home/jtong/lbyl/data/cep/"
        return [os.path.join(cep_dir, "AOD.41014010._000001.pool.root.1") # mc23_5p36TeV.860225.SuperChic_4p2_CEPincohgammagamma_2Mgg500.merge.AOD.e8558_e8528_s4357_s4258_r15716_r14663_tid41014010_00
                ]
    elif s == "alp":
        alp_dir = "/home/jtong/lbyl/data/alp/"
        return [os.path.join(alp_dir, "AOD.41382354._000001.pool.root.1"), # mc23_5p36TeV.860228.SuperChic_4p2_axion_ma6_v2.merge.AOD.e8558_e8528_s4357_s4258_r15716_r15516_tid41382354_00
                os.path.join(alp_dir, "AOD.41382338._000001.pool.root.1"), # mc23_5p36TeV.860227.SuperChic_4p2_axion_ma5_v2.merge.AOD.e8558_e8528_s4357_s4258_r15716_r15516_tid41382338_00
                ]
    else:
        raise ValueError(f"Unknown sample type: {sample}")

# =====================================
# Event processing (cutflow + histograms + NN outputs)
# =====================================
def process_events(t, aco_mode="cr"):
    
    print("Initializing tools...")

    #1) Trigger Decision Tool
    #=== Setup xAODConfigTool ===
    print("initializing TrigConfigTool")
    configTool = ROOT.TrigConf.xAODConfigTool("TrigConfigTool")
    if not configTool.initialize().isSuccess():
        print("Failed to initialize TrigConfigTool")
        return
    
    #if TrigDecisionTool is None:
    print("Initializing TrigDecisionTool...")
    TrigDecisionTool = ROOT.Trig.TrigDecisionTool("TrigDecisionTool")
    TrigDecisionTool.setProperty("ConfigTool", "TrigConfigTool").ignore()
    TrigDecisionTool.setProperty("TrigDecisionKey", "xTrigDecision").ignore()
    TrigDecisionTool.setProperty("AcceptMultipleInstance", True).ignore()
    TrigDecisionTool.initialize()
    print(" ---- TrigDecisionTool ready")


    # 2) EG Calibration and Smearing Tool
    #if egtool is None:
    print("Initializing EG Tool...")
    egtool = ROOT.AtlasRoot.egammaEnergyCorrectionTool()
    egtool.setESModel(ROOT.egEnergyCorr.es2017_R21_ofc0_v1)
    egtool.initialize()
    print(" ---- EG Tool ready")

    #if EGCalibrationAndSmearingTool is None:
    print("Initializing EG Calibration and Smearing Tool...")
    EGCalibrationAndSmearingTool = ROOT.CP.EgammaCalibrationAndSmearingTool("EGCalibrationAndSmearingTool")
    EGCalibrationAndSmearingTool.setProperty("ESModel", "es2017_R21_ofc0_v1").ignore()
    EGCalibrationAndSmearingTool.setProperty("useFastSim", 0).ignore()
    EGCalibrationAndSmearingTool.setProperty("decorrelationModel", "1NP_v1").ignore()
    EGCalibrationAndSmearingTool.setProperty("randomRunNumber", "123456").ignore()
    EGCalibrationAndSmearingTool.setProperty("FixForMissingCells", True).ignore()
    EGCalibrationAndSmearingTool.initialize()
    print(" ---- EG Calibration and Smearing Tool done")
    
    # --- cutflow counters ---
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
    # --- acoplanrity scan ---
    ACO_SCAN = {f"aco<={s:.3f}": 0 for s in np.arange(0.005, 0.015, 0.001)}
    
    # --- diphoton kinematic lists (UN‑WEIGHTED) ---
    diphoton_masses        = []
    diphoton_pts           = []
    diphoton_acoplanarity  = []
    leading_photon_ets     = []
    diphoton_rapidity_diff = []
    diphoton_cos_thetas    = []
    
    # Per‑event weight (one‑to‑one with arrays above)
    event_weights = []
    
    # --- new photon-level arrays ---
    output_photon_pt       = []
    output_photon_eta      = []
    output_photon_idNN_out = []
    output_photon_idNN_pass= []

    total_entries = t.GetEntries()
    for i in range(total_entries):
        t.GetEntry(i)
        cut_flow["Total Events"] += 1

        # 1) Trigger requirement
        if not (
            TrigDecisionTool.isPassed("HLT_mb_sp_vpix30_hi_FgapAC5_L1TAU1_TE4_VTE200") or
            TrigDecisionTool.isPassed("HLT_mb_sp_vpix30_hi_FgapAC5_L12TAU1_VTE200") or
            TrigDecisionTool.isPassed("HLT_mb_sp_vpix30_hi_FgapAC5_2g0_etcut_L12TAU1_VTE200")
        ):
            continue

        # 2) Photon selection + calibration + DNN PID
        photons_eta = []
        photons_pt  = []
        selected_photons = []
        for ph in t.Photons:
            # a) apply EG calibration & smearing
            my_photon = ROOT.xAOD.Photon( ph )
            EGCalibrationAndSmearingTool.applyCorrection(my_photon)

            pt   = ph.pt()
            eta  = ph.eta()
            aeta = abs(ph.caloCluster().etaBE(2))
            goodOQ = ph.isGoodOQ(ROOT.xAOD.EgammaParameters.BADCLUSPHOTON)
            pass_eta = (aeta < 2.47 and not (1.37 < aeta < 1.52))
            pass_pt  = (pt > 2.5 * GeV)
            if pass_eta: photons_eta.append(ph)
            if pass_pt:  photons_pt.append(ph)
            if not (pass_eta and pass_pt and goodOQ):
                continue

            # b) read shower-shape inputs
            # f1     = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.f1)
            # eratio = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.Eratio)
            # fracs1 = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.fracs1)
            # e277   = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.e277)
            # f1core = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.f1core)
            f1     = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.f1)
            weta2 = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.Eratio)
            eratio = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.fracs1)
            fracs1   = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.e277)
            f1core = ph.showerShapeValue(ROOT.xAOD.EgammaParameters.f1core)


            # c) pick eta‐category for the DNN
            aeta = abs(ph.eta())
            cat = 0
            if aeta < 0.6:       cat = 1
            elif aeta < 1.37:    cat = 2
            elif aeta < 1.81:    cat = 3
            else:                cat = 4

            reader, vars_tuple = get_nn_reader(cat)
            # fill the reader's c_float slots
            #for val, slot in zip((f1, eratio, fracs1, e277, f1core), vars_tuple):
            for val, slot in zip((f1, weta2, eratio, fracs1, f1core), vars_tuple):
                slot.value = val
            # compute the DNN score
            score   = reader.EvaluateMVA("Signal")
            cut_val = [0, 0.45, 0.5, 0.3, 0.5][cat]
            nn_pass = (score > cut_val)

            # d) store photon‐level outputs (in GeV)
            output_photon_pt.append(pt / GeV)
            output_photon_eta.append(eta)
            output_photon_idNN_out.append(score)
            output_photon_idNN_pass.append(nn_pass)

            # only NN‐passed photons are used in diphoton selection
            if nn_pass:
                selected_photons.append(ph)

        # photon‐cut bookkeeping
        if len(photons_eta) >= 2: cut_flow["Pass Photon eta cut"] += 1
        if len(photons_pt)  >= 2: cut_flow["Pass Photon pt cut"] += 1
        if len(selected_photons) < 2:
            continue
        cut_flow["Pass Photon Selection"] += 1

        # 3) Diphoton mass cut
        selected_photons.sort(key=lambda x: x.pt(), reverse=True)
        lead, sub = selected_photons[:2]
        m = invariant_mass(lead, sub) / GeV
        if m < 5:
            continue
        cut_flow["Pass Diphoton Selection"] += 1

        ##################################################################################################
        # Veto
        ##################################################################################################

        # 4) Track veto
        good_tracks_for_veto = []
        for track in t.InDetTrackParticles:
            eta = track.eta()
            pt = track.pt()
            passesTrackKinematicPreSelection = (pt > 0.1 * GeV and abs(eta) < 2.5)
            npix = ctypes.c_ubyte(0)
            nsct = ctypes.c_ubyte(0)
            nBL = ctypes.c_ubyte(0)
            nIBL = ctypes.c_ubyte(0)
            nPixHole = ctypes.c_ubyte(0)
            track.summaryValue(npix, ROOT.xAOD.SummaryType.numberOfPixelHits)
            track.summaryValue(nsct, ROOT.xAOD.numberOfSCTHits)
            track.summaryValue(nBL, ROOT.xAOD.numberOfBLayerHits)
            track.summaryValue(nIBL, ROOT.xAOD.numberOfInnermostPixelLayerHits)
            track.summaryValue(nPixHole, ROOT.xAOD.numberOfPixelHoles)
            npix_val = npix.value
            nsct_val = nsct.value
            nBL_val = nBL.value
            nIBL_val = nIBL.value
            nPixHole_val = nPixHole.value
            passesTrackPreSelection = (npix_val >= 1 and (npix_val + nsct_val) >= 6)
            if passesTrackKinematicPreSelection and passesTrackPreSelection:
                good_tracks_for_veto.append(track)

        if len(good_tracks_for_veto) > 0:
            continue
        cut_flow["Pass Track Veto"] += 1


        # 5) Pixel track veto
        n_pixel_tracks = 0
        for track in t.InDetPixelTrackParticles:
            eta = track.eta()
            pt = track.pt()
            npix = ctypes.c_ubyte(0)
            track.summaryValue(npix, ROOT.xAOD.SummaryType.numberOfPixelHits)
            npix_val = npix.value
            if pt > 0.05 * GeV and abs(eta) < 2.5 and npix_val >= 3:
                for ph in t.Photons:
                    ph_eta = ph.eta()
                    ph_aeta = abs(ph_eta)
                    ph_pt = ph.pt() / GeV
                    if ph_aeta < 2.37 and not (1.37 < ph_aeta < 1.52) and ph_pt > 0.5 * GeV:
                        if abs(ph_eta - eta) < 0.5:
                            n_pixel_tracks += 1
                            break
        if n_pixel_tracks > 0:
            continue
        cut_flow["Pass Pixel Track Veto"] += 1
        
        
        # 6) Acoplanarity 
        dphi = abs(lead.phi() - sub.phi())
        if dphi > math.pi: dphi = 2*math.pi - dphi
        A_phi = 1 - (dphi / math.pi)
        
        # Apply the acoplanarity selection according to aco_mode
        if aco_mode == "cr":
            # keep events with A_phi ≥ 0.01
            if A_phi <= 0.01:
                continue
        elif aco_mode == "sr":
            # keep events with A_phi ≤ 0.01
            if A_phi >= 0.01:
                continue
        elif aco_mode == "sr_scan":     # **NEW – working scan**
            #
            # fill the running counters first …
            #
            for aco_thr in np.arange(0.005, 0.015, 0.001):
                if A_phi >= aco_thr:
                    continue
                ACO_SCAN[f"aco<={aco_thr:.3f}"] += 1
            
        cut_flow["Pass Acoplanarity Selection"] += 1

        cluster1 = lead.caloCluster()
        cluster2 = sub.caloCluster()
        l1_weight = l1_trigger_weight(cluster1, cluster2, (5.63188, 1.63362, 0.223143))
        sf_weight = scale_factor_weight(lead, sub, sf_hist)

        if len(good_tracks_for_veto) >= 2:
            good_tracks_for_veto.sort(key=lambda x: x.pt(), reverse=True)
            hlt_wt = hlt_weight(good_tracks_for_veto[0], good_tracks_for_veto[1], (-0.00471974, 0, 0.993623))
        else:
            hlt_wt = 1.0

        spacepoints = []
        for track in good_tracks_for_veto:
            radius = track.radiusOfFirstHit()
            phi = track.phi()
            eta = track.eta()
            x = radius * math.cos(phi)
            y = radius * math.sin(phi)
            z = radius * math.sinh(eta)
            vec = ROOT.TVector3(x, y, z)
            spacepoints.append(vec)

        for sp in spacepoints:
            if passes_bl_spacepoint_preselection(sp, [lead, sub]):
                print("Photon matched to spacepoint")

        total_weight = l1_weight * hlt_wt * sf_weight

        # Build TLorentzVectors for final kinematics
        p1 = ROOT.TLorentzVector(); p2 = ROOT.TLorentzVector()
        p1.SetPtEtaPhiM(lead.pt(), lead.eta(), lead.phi(), 0)
        p2.SetPtEtaPhiM(sub.pt(),  sub.eta(),  sub.phi(),  0)
        pt_sum = (p1 + p2).Pt() / GeV



        diphoton_masses.append(m)
        diphoton_pts.append(pt_sum)
        diphoton_acoplanarity.append(A_phi)

        # Leading photon E_T
        e_lead = p1.E()
        et_lead = e_lead / math.cosh(p1.Eta())
        leading_photon_ets.append((et_lead / GeV))

        # Rapidity difference and |cosθ|
        dy = p1.Rapidity() - p2.Rapidity()
        diphoton_rapidity_diff.append(dy)
        diphoton_cos_thetas.append((abs(math.tanh(dy/2.0))) )
        
        event_weights.append(total_weight)

        # Electron selection
        n_electrons = 0
        for el in t.Electrons:
            el_pt = el.pt()
            el_quality = el.isGoodOQ(ROOT.xAOD.EgammaParameters.BADCLUSELECTRON)
            el_eta = abs(el.caloCluster().etaBE(2))
            if el_quality and el_pt > 2 * GeV and el_eta < 2.47 and (el_eta > 1.52 or el_eta < 1.37):
                n_electrons += 1
        cut_flow["Pass Electron Selection"] += n_electrons   

    return {
        "cut_flow": cut_flow,
        "diphoton_masses":        diphoton_masses,
        "diphoton_pts":           diphoton_pts,
        "diphoton_acoplanarity":  diphoton_acoplanarity,
        "leading_photon_ets":     leading_photon_ets,
        "diphoton_rapidity_diff": diphoton_rapidity_diff,
        "diphoton_cos_thetas":    diphoton_cos_thetas,
        "event_weights":          event_weights,
        # photon-level outputs
        "photon_pt":        output_photon_pt,
        "photon_eta":       output_photon_eta,
        "photon_idNN_out":  output_photon_idNN_out,
        "photon_idNN_pass": output_photon_idNN_pass,
    }

# =====================================
# Cutflow printer
# =====================================
def print_cutflow(sample_label, cut_flow):
    print(f"\nCutflow for {sample_label}:")
    total = cut_flow.get("Total Events", 0)
    order = [
        ("All events", "Total Events"),
        ("Photon eta cut",   "Pass Photon eta cut"),
        ("Photon pt cut",    "Pass Photon pt cut"),
        ("Photon selection", "Pass Photon Selection"),
        ("Diphoton selection","Pass Diphoton Selection"),
        ("Track veto",       "Pass Track Veto"),
        ("Pixel veto",       "Pass Pixel Track Veto"),
        ("Electron sel.",    "Pass Electron Selection"),
        ("Acoplanarity sel.", "Pass Acoplanarity Selection"),
    ]
    for label, key in order:
        passed = cut_flow.get(key, 0)
        if key == "Pass Electron Selection":
            print(f"{label:25s} {passed:7d}")
        else:
            eff = (passed/total*100) if total else 0
            err = (100*math.sqrt(eff/100*(1-eff/100)/total)) if total else 0
            print(f"{label:25s} {passed:7d}  {eff:6.2f}%+/-{err:.2f}%")
