import pickle
import os
import numpy as np

path = '/home/jtong/lbyl/Plots_cr/histograms_cr.pkl'

if not os.path.exists(path):
    print(f'File not found at {path}. Please verify the path and ensure it is accessible.')
else:
    with open(path, 'rb') as f:
        histograms = pickle.load(f)
        #print(histograms.keys())

    for sample in ['run2data', 'signal', 'yyee', 'cep']:
        try:
            counts = histograms[sample]['mass']['counts']
            print(f'\n{sample} mass counts:')
            print(counts)
            print('sum of counts = ', np.sum(counts))
        except KeyError as err:
            print(f'\nData for sample "{sample}" could not be found: {err}')
