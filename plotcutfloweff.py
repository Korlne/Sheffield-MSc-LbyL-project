import numpy as np
from plots import plot_cutflows, load_cutflow_data
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLASAlt)

data = load_cutflow_data("cr")
plot_cutflows(data, plot_dir="Plots_cr")
plt.show() 