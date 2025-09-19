#!/usr/bin/env python
import ROOT
import pickle
import os
import glob
import numpy as np
from AthenaCommon.SystemOfUnits import nanobarn

# Paths to the YY→ee ROOT files
root_files = [
    '/home/jtong/lbyl/run2data_binned/user.slawlor.Lbylntuples.periodAllYear2.DAOD_HION4.grp23_v01_p6774_198765_Smc5000_Sppc2500.root',
]

# Directory under 'Nominal' after applying all cuts including AcoCut
#hist_dir_key = (
#    'CutFlow_Grl_Trigger_OotPileUp_StandardTrackVeto_BlSpacePointVeto_PixelTrackVeto_PtCut_MCut_AcoCut'
#)
hist_dir_key = ('3_LbLeventselection')
# Histogram names to extract post-AcoCut
hist_names = [
    'h_ZMassZoom',
    'h_ZptZoom',
    'h_Zrapidity',
    'h_ZAcoZoom',
    'h_ZLeadingPhotonET'
]

'''
# Cross sections per sample token (in nanobarn)
scaling_factors = {
    'M5': 7780 * nanobarn,
    'M6': 210000 * nanobarn,
    '4M7': 750000 * nanobarn,
    '3M4': 938097 * nanobarn
}
'''

def extract_histogram_info(hist):
    '''Return edges, counts, and errors arrays from a ROOT TH1.'''
    n_bins = hist.GetNbinsX()
    edges = [hist.GetBinLowEdge(i) for i in range(1, n_bins+1)]
    edges.append(hist.GetBinLowEdge(n_bins) + hist.GetBinWidth(n_bins))
    counts = [hist.GetBinContent(i) for i in range(1, n_bins+1)]
    errors = [hist.GetBinError(i)   for i in range(1, n_bins+1)]
    return np.array(edges), np.array(counts), np.array(errors)


def process_root_files(files):
    '''Extract all histograms post-AcoCut from each file, scale them, and pickle.'''  
    dataLumi = 1.67 / nanobarn
    total_mcEv = 0 # Initialise
    for filepath in files:
        basename = os.path.basename(filepath)
        token = basename.split('.')[3]
        print(f'Processing file {basename} (token={token})')

        f = ROOT.TFile.Open(filepath)
        if not f or f.IsZombie():
            print('  ERROR: cannot open ROOT file')
            continue

        # path prefix under 'Nominal'
        prefix = f'Nominal/{hist_dir_key}/Pair'
        hist_info = {}
        # extract all histograms
        for name in hist_names:
            h = f.Get(f'{prefix}/{name}')
            if not h:
                print(f'  WARNING: {name} not found under {prefix}')
                continue
            edges, counts, errors = extract_histogram_info(h)
            hist_info[name] = {'edges': edges, 'counts': counts, 'errors': errors}
        '''
        # determine scale from MC events
        tot = f.Get('eventVeto/TotalEvents')
        mcEv = tot.GetBinContent(2) if tot else 0
        cross = scaling_factors.get(token, 1.0)
        scale = (dataLumi * cross / mcEv) if mcEv else 1.0
        print(f'  Scale factor = {scale}')
        total_mcEv += mcEv
        
        # apply scaling
        for d in hist_info.values():
            d['counts'] *= scale
            d['errors'] *= scale
        '''
        # write out pickle: alp_root_<token>_hist.pkl
        outname = os.path.join(os.path.dirname(filepath), f'run2data_root_{token}_hist.pkl')
        with open(outname, 'wb') as fo:
            pickle.dump(hist_info, fo)
        print(f'  Saved pickle {outname}\n')
    # report total MC events processed
    # print(f'Total MC events processed across all alp samples: {total_mcEv}')

def merge_histograms(pkl_dir, hist_names):
    '''Merge all per-file pickles into one composite dictionary with elementwise addition.'''  
    files = glob.glob(os.path.join(pkl_dir, 'run2data_root_*_hist.pkl'))
    # Exclude merged output if present
    files = [pf for pf in files if 'merged' not in os.path.basename(pf)]
    merged = {name: None for name in hist_names}

    for pf in files:
        data = pickle.load(open(pf, 'rb'))
        for name in hist_names:
            d = data.get(name)
            
            if d is None:
                continue
            # convert to numpy for elementwise addition
            edges = np.array(d['edges'])
            counts = np.array(d['counts'], float)
            errors = np.array(d['errors'], float)
            #print(f"{pf} → {name}: counts shape = {counts.shape}")
            
            if merged[name] is None:
                merged[name] = {'edges': edges.copy(),
                                'counts': counts.copy(),
                                'errors': errors.copy()}
                #print(f"After merge: {name} counts length = {merged[name]['counts'].shape}")
            else:
                # binning same; add counts, combine errors in quadrature
                merged[name]['counts'] += counts
                merged[name]['errors'] = np.sqrt(merged[name]['errors']**2 + errors**2)
                #print(f"After merge: {name} counts length = {merged[name]['counts'].shape}")
                
    
    return merged

if __name__ == '__main__':
    # 1) Process and pickle all post-AcoCut histograms per sample
    process_root_files(root_files)

    # 2) Merge into one pickle
    pkl_dir = os.path.dirname(root_files[0])
    merged = merge_histograms(pkl_dir, hist_names)
    merged_out = os.path.join(pkl_dir, 'run2data_root_merged_hist.pkl')
    
    # Convert numpy arrays to lists for stable pickling
    for d in merged.values():
        d['edges'] = d['edges'].tolist()
        d['counts'] = d['counts'].tolist()
        d['errors'] = d['errors'].tolist()
    with open(merged_out, 'wb') as fo:
        pickle.dump(merged, fo)
    print(f'Saved merged pickle: {merged_out}')
    
    # Print total summed events from four YY→ee sources for each histogram
    print("=== Total events in merged histograms ===")
    for name in hist_names:
        total = sum(merged[name]['counts'])
        print(f"{name}: {total}")

