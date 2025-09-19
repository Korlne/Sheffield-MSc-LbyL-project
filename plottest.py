import numpy
import pickle


pickle_path = '/home/jtong/lbyl/cutflow_alp40_aco-cr.pkl'
with open(pickle_path, "rb") as f:
        hist_pickle = pickle.load(f)
        
print(hist_pickle['diphoton_masses'])
print('*'*50)


