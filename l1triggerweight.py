import pickle, numpy as np, pathlib

pkl = pathlib.Path("cutflow_signal_aco-cr.pkl")
with pkl.open("rb") as f:
    sig = pickle.load(f)

print("Keys in pickle:", sig.keys())          # <- ‘event_weights’ should be listed
w = np.asarray(sig.get("event_weights", []))
print("N events           :", len(w))
#print("min / max weight   :", w.min(), w.max())
#print("fraction that are 1:", np.mean(w == 1))
print("5-sample preview   :", w[:5])
