#!/usr/bin/env python
import os
import pickle
import glob
import numpy as np

# Directory where your pickles live
pkl_dir = '/home/jtong/lbyl/yyee_binned'

# Per-sample files: yyee_root_<token>_hist.pkl
sample_pickles = glob.glob(os.path.join(pkl_dir, 'yyee_root_*_hist.pkl'))
# Exclude the merged file if it’s in the same folder
sample_pickles = [p for p in sample_pickles if 'merged' not in p]

# Merged pickle
merged_pickle = os.path.join(pkl_dir, 'yyee_root_merged_hist.pkl')

# Load merged counts
with open(merged_pickle, 'rb') as f:
    merged = pickle.load(f)
merged_aco = np.array(merged['h_ZAcoZoom']['counts'], float)

print("=== h_ZAcoZoom per-sample counts ===")
for pkl in sorted(sample_pickles):
    token = os.path.basename(pkl).split('_')[2]  # extracts e.g. "20M", "7M20", etc.
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    counts = np.array(data['h_ZAcoZoom']['counts'], float)
    print(f"{token:>5}: {counts.tolist()}")

print("\n=== h_ZAcoZoom merged counts ===")
print(merged_aco.tolist())

# Verify elementwise sum equals merged
summed = sum(
    np.array(pickle.load(open(pkl, 'rb'))['h_ZAcoZoom']['counts'], float)
    for pkl in sample_pickles
)
print("\n=== element-wise sum of per-sample counts ===")
print(summed.tolist())

diff = merged_aco - summed
print("\n=== merged minus summed (should all be 0) ===")
print(diff.tolist())
