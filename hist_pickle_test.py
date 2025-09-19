import pickle
import numpy as np

# load whichever file you wrote, e.g.  Plots_sr/histograms_sr.pkl
with open("Plots_cr/histograms_cr.pkl", "rb") as f:
    h = pickle.load(f)

# pick the sample and variable you want
sample   = "signal"      # or "signal", "alp5", …
variable = "mass"     # one of: "mass", "pt", "acop", "leading_et"

counts = np.sum(h[sample][variable]["counts"])
print(counts)