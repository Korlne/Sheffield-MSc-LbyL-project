import argparse, pickle, numpy as np, os, re

parser = argparse.ArgumentParser(
    description="Find optimal ALP mass windows (uses yyee bins).")
parser.add_argument("alp_file", help="Path to the ALP pickle file")
args = parser.parse_args()

# ------------------------------------------------------------------- #
# 1.  Inputs                                                          #
# ------------------------------------------------------------------- #
p_alp   = args.alp_file                                         # signal
p_yyee  = "/home/jtong/lbyl/yyee_binned/yyee_root_merged_hist.pkl"  # bkg
p_cep   = "cutflow_cep_aco-sr.pkl"                                   # bkg
p_lbyl  = "cutflow_sig_aco-sr.pkl"                                   # bkg

# Adjust the overall normalisations (from plots.py) ------------------ #
scale_factors = {
    "alp" : 1.00,
    "yyee": 1.00,
    "cep" : 1.00,
    "lbyl": 1.00,
}

# Load every sample into a single dict so `build_hist` can see it ----- #
data = {
    "alp" : pickle.load(open(p_alp , "rb")),
    "yyee": pickle.load(open(p_yyee, "rb")),
    "cep" : pickle.load(open(p_cep , "rb")),
    "lbyl": pickle.load(open(p_lbyl, "rb")),
}

# ------------------------------------------------------------------- #
# 3.  The histogram builder                                           #
# ------------------------------------------------------------------- #
def build_hist(sample, variable, bins):
    """Return (counts, bins) for the requested variable."""
    scale = scale_factors[sample]

    if sample == "yyee":                     # pre‑merged histograms
        key_map = {
            "mass":       "h_ZMassZoom",
            "pt":         "h_ZptZoom",
            "acop":       "h_ZAcoZoom",
            "leading_et": "h_ZLeadingPhotonET",
        }
        obj = data[sample][key_map[variable]]
        return np.asarray(obj["counts"]) * scale, bins

    # MC samples stored as per‑event arrays + weights ---------------- #
    results = data[sample]
    arrays = {
        "mass":          results["diphoton_masses"],
        "pt":            results["diphoton_pts"],
        "acop":          results["diphoton_acoplanarity"],
        "leading_et":    results["leading_photon_ets"],
        "rapidity_diff": results["diphoton_rapidity_diff"],
        "costheta":      results["diphoton_cos_thetas"],
    }
    vals = np.asarray(arrays[variable])
    w    = np.asarray(results["event_weights"]) * scale
    counts, _ = np.histogram(vals, bins=bins, weights=w)
    return counts, bins

# ------------------------------------------------------------------- #
# 4.  Choose the variable and common binning                          #
# ------------------------------------------------------------------- #
VAR  = "mass"
bins = np.asarray(data["yyee"]["h_ZMassZoom"]["edges"])  # always yyee bins

sig_counts, _  = build_hist("alp" , VAR, bins)
yyee_counts, _ = build_hist("yyee", VAR, bins)
cep_counts , _ = build_hist("cep" , VAR, bins)
lbyl_counts, _ = build_hist("lbyl", VAR, bins)

bkg_counts = yyee_counts + cep_counts + lbyl_counts

# ------------------------------------------------------------------- #
# 5.  Fake data spectrum                                              #
# ------------------------------------------------------------------- #
rng = np.random.default_rng(seed=42)
data_counts = rng.poisson(sig_counts + bkg_counts)

mass_centres = 0.5*(bins[:-1] + bins[1:])       # bin centres

# ------------------------------------------------------------------- #
# 6.  Cumulative sums                                                 #
# ------------------------------------------------------------------- #
sig_cum = np.concatenate(([0], np.cumsum(sig_counts)))
bkg_cum = np.concatenate(([0], np.cumsum(bkg_counts)))
dat_cum = np.concatenate(([0], np.cumsum(data_counts)))

def integral(cum, lo, hi):
    """Integral between inclusive indices lo … hi‑1 (same as hist bins)."""
    return cum[hi] - cum[lo]

# ------------------------------------------------------------------- #
# 7.  Tightest ≥80 % signal window                                    #
# ------------------------------------------------------------------- #

def optimal_window(i_center, frac_sig=0.80):
    tot_sig = sig_counts[i_center]
    if tot_sig == 0:
        return i_center, i_center + 1

    lo = hi = i_center
    sig_in = sig_counts[i_center]

    while sig_in < frac_sig * tot_sig:
        gain_lo = sig_counts[lo - 1] if lo > 0 else 0
        gain_hi = sig_counts[hi]     if hi < len(sig_counts) - 1 else 0
        if gain_lo >= gain_hi and lo > 0:
            lo -= 1;  sig_in += gain_lo
        elif hi < len(sig_counts) - 1:
            hi += 1;  sig_in += gain_hi
        else:
            break
    return lo, hi + 1  # hi is exclusive

# ------------------------------------------------------------------- #
# 8.  Enforce ≥1 background event                                     #
# ------------------------------------------------------------------- #

def enlarge_until_background(lo, hi, min_bkg=1.0):
    while integral(bkg_cum, lo, hi) < min_bkg:
        add_left  = bkg_counts[lo - 1] if lo > 0 else np.inf
        add_right = bkg_counts[hi]     if hi < len(bkg_counts) else np.inf
        if add_left <= add_right and lo > 0:
            lo -= 1
        elif hi < len(bkg_counts):
            hi += 1
        else:
            break
    return lo, hi

# ------------------------------------------------------------------- #
# 9.  Scan every mass point                                           #
# ------------------------------------------------------------------- #
records = []
for i_c, m in enumerate(mass_centres):
    lo, hi = optimal_window(i_c)            # tight ≥80 % signal
    lo, hi = enlarge_until_background(lo, hi)
    records.append({
        "m [GeV]"  : m,
        "lo [GeV]" : bins[lo],
        "hi [GeV]" : bins[hi],
        "width"    : bins[hi] - bins[lo],
        "s_exp"    : integral(sig_cum, lo, hi),
        "b_exp"    : integral(bkg_cum, lo, hi),
        "n_obs"    : integral(dat_cum, lo, hi),
    })

# ------------------------------------------------------------------- #
# 10. Save the results                                                #
# ------------------------------------------------------------------- #
# Extract the ALP mass from the input filename – look for ‘XmYYGeV’ or
# a plain number followed by “GeV”; fall back to the filename stem.
basename = os.path.basename(p_alp)
match = re.search(r"(\d+\.?\d*)\s*[gG][eE][vV]", basename)
alp_mass_tag = match.group(1) if match else os.path.splitext(basename)[0]

out_file = f"windows_{alp_mass_tag}.pkl"
with open(out_file, "wb") as f_out:
    pickle.dump(records, f_out)

# Quick sanity display ---------------------------------------------- #
print("First five windows:")
for rec in records[:5]:
    print(rec)
print(f"\nSaved {len(records)} windows to '{out_file}'.")
