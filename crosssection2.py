import pickle
import numpy as np
from numpy.random import default_rng
from significance_analysis_root4 import pull_profiled_shape
from pathlib import Path
# ----------------------------
# Inputs
# ----------------------------
pkl_file = Path("pickle_Sean/bkg_alp_sr_pickle.pkl")
lumi = 1.63  # nb^-1
# eff = 3547/5733  # efficiency (fraction)
eff = 3937/5866
N_obs = 56.42   # observed count
# Load pickle
with open(pkl_file, "rb") as f:
    bkg_pkl = pickle.load(f)
# Check available backgrounds
print("Top-level keys:", bkg_pkl.keys())
if "cep" not in bkg_pkl or "yy2ee" not in bkg_pkl:
    raise KeyError("Background pickle missing 'cep' or 'yy2ee' keys.")
# Inspect histogram names in nominal dicts
cep_nominal_keys = bkg_pkl["cep"]["nominal"].keys()
yy2ee_nominal_keys = bkg_pkl["yy2ee"]["nominal"].keys()
print("CEP nominal keys:", cep_nominal_keys)
print("yy2ee nominal keys:", yy2ee_nominal_keys)
print("CEP systematics keys:", bkg_pkl["cep"]["systematics"].keys())
print("yy2ee systematics keys:", bkg_pkl["yy2ee"]["systematics"].keys())
# Auto-select hist_name if available
if "h_ZMassFine" in cep_nominal_keys:
    hist_name = "h_ZMassFine"
else:
    raise KeyError("Expected 'h_ZMassFine' histogram not found in background pickle.")
rng = default_rng()
# ----------------------------
# Profiled background (systematics pulled)
# ----------------------------
# bkg_cep = pull_profiled_shape(bkg_pkl["cep"], hist_name, rng)
# bkg_yy2ee = pull_profiled_shape(bkg_pkl["yy2ee"], hist_name, rng)
# N_bkg = (bkg_cep + bkg_yy2ee).sum()
bkg_cep = 14.38
bkg_yy2ee = 13.54
#N_bkg = bkg_cep+bkg_yy2ee
N_bkg = 0
# ----------------------------
# Cross-section calculation
# ----------------------------
sigma = (N_obs - N_bkg) / (lumi * eff)
# Statistical uncertainty
stat_err = np.sqrt(N_obs + N_bkg) / (lumi * eff)
# Systematic uncertainty via profiling spread
n_toys = 10000
sigma_toys = []
for _ in range(n_toys):
    bkg_cep_toy = pull_profiled_shape(bkg_pkl["cep"], hist_name, rng)
    bkg_yy2ee_toy = pull_profiled_shape(bkg_pkl["yy2ee"], hist_name, rng)
    N_bkg_toy = (bkg_cep_toy + bkg_yy2ee_toy).sum()
    sigma_toys.append((N_obs - N_bkg_toy) / (lumi * eff))
syst_err = np.std(sigma_toys)
print(f"σ = {sigma:.2f} ± {stat_err:.2f} (stat) ± {syst_err:.2f} (syst) nb")
print('Number of cep', bkg_cep)
print('Number of yyee', bkg_yy2ee)