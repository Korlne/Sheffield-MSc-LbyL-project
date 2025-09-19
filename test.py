import pickle, numpy as np, pathlib

# pkl = pathlib.Path("/home/jtong/lbyl/yyee_binned/yyee_root_merged_hist_aco-cr.pkl")

# with pkl.open("rb") as f:
#     yyee = pickle.load(f)

# edges = np.asarray(yyee["h_ZMassZoom"]["edges"], dtype=float)
# print("yyee mass-bin edges:", edges)
# print("Number of bins:", len(edges) - 1)
# print("Bin centres:", (edges[:-1] + edges[1:]) / 2)


pkl = pathlib.Path("yyee_binned/yyee_root_merged_hist_aco-sr.pkl")

with pkl.open("rb") as f:
     histpkl = pickle.load(f)

print(histpkl.items()) 