# Script for generating a dictionary with all the relavant information of the GYRE models.
# If you use the provided grids in /grids/gyre_per_l1m*_ext.pkl, this script is not needed.
# Joey Mombarg - 19 Oct 2023


import numpy as np
import glob
import pickle, h5py

def read_GYRE_summary(filename, n_low = -150, n_high = 0):
    '''
    Read a GYRE h5 file and return the frequencies and radial order of the requested range between n_low and n_high.
    '''
    q = h5py.File(filename, "r")
    n_pg_ = np.array(q['n_pg'][:])
    freq_ = np.array([a[0] for a in q['freq'][:]])
    n_g_  = np.array(q['n_g'][:])
    get = np.array([n_pg_ >= n_low]) & np.array([n_pg_ <= n_high])
    get = get[0]
    periods = 1 / np.array(freq_[get])
    return np.array(n_pg_[get]), np.array(freq_[get]), np.array(n_g_[get])

gyre_per = {}
gyre_per['model'] = []
# Create a directory where gyre_per['n'] contains the periods of all models for radial order n.
for ng_ in np.arange(1,101,1):
    gyre_per[str(ng_)] = []

for gf in glob.glob('GRID_SHERLOCK/*/g-modesl1m-1*/summary*'):
    n_pg, freq, ng = read_GYRE_summary(gf)
    model = gf.split('/')[1]
    gyre_per['model'].append(model)
    for ng_ in np.arange(1,101,1):
        if ng_ in ng:
            ing = np.where(ng == ng_)[0][0]
            if 1/np.abs(freq[ing]) > 0.5 and 1/np.abs(freq[ing]) < 10:
                gyre_per[str(ng_)].append(1/np.abs(freq[ing]))
            else:
                gyre_per[str(ng_)].append(np.nan)
        else:
            gyre_per[str(ng_)].append(np.nan)

# Save as pickle file.
with open('gyre_per_l1m-1_ext.pkl', 'wb') as f:
    pickle.dump(gyre_per, f)
