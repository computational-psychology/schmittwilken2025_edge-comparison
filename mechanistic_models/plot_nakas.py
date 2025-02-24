"""
This script can be used to plot the fitted Naka-Rushton functions of the
mechanistic models.

In addition, we plot the distribution of filter activities for the
models. This is useful to understand the relevant range of the Naka-Rushton-
function.
To get the distributions of filter activations, we simply run the spatial
(and temporal) filters of the models on all possible input stimuli.

@author: Lynn Schmittwilken, Feb 2025
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from time import time
import pickle

from functions import create_filter_outputs, create_edge, create_noises, \
    load_params, naka_rushton

sys.path.insert(1, '../experimental_data')
from exp_params import stim_params as sparams

np.random.seed(0)

nInstances = 10                     # number of noise instances used for testing
xmax = 5
edges = np.linspace(0, xmax, 100)   # bins for histogram


####################################
#            Read data             #
####################################
df = pd.read_csv("../experimental_data/expdata_pooled.txt", sep=" ")
noise_conds = np.unique(df["noise"])   # Noise conditions
edge_conds = np.unique(df["edge"])     # Edge conditions


def get_activity(params, mparams, e, noiseList, c):
    edge = create_edge(c, e, sparams)
    for t in range(mparams["n_trials"]):
        mout1, _ = create_filter_outputs(edge, noiseList[t], noiseList[t-1], mparams)
        H1, _ = np.histogram(mout1.flatten(), bins=edges, density=True)
    return H1 / H1.sum()


def get_activities(fname, dfPool, noiseDict, edges, load=True):
    outName = fname.removesuffix(".pickle") + '_activations.pickle'
    best_params, mparams = load_params(fname)
    mparams["n_trials"] = nInstances
    
    if load:
        try:
            # Load data from pickle:
            with open(outName, 'rb') as handle:
                data_pickle = pickle.load(handle)
                acts = data_pickle["acts"]
                edges = data_pickle["edges"]
        except:
            print("Could not load activations.")
            load = False

    if not load:
        acts = np.zeros(len(edges)-1)
        for ni, n in enumerate(noise_conds): # [noise_conds!="none"]
            print(n)
            for ei, e in enumerate(edge_conds[::-1]):
                # Get performance of average observer (all contrasts)
                dfTemp = dfPool[(dfPool["noise"]==n) & (dfPool["edge"]==e)]
    
                # Loop through all contrasts
                for i, c in enumerate(dfTemp["contrasts"].to_numpy()):
                    acts += get_activity(best_params, mparams, e, noiseDict[n], c)
        
        # Save pickle
        with open(outName, 'wb') as handle:
            pickle.dump({"acts": acts, "edges": edges}, handle)

    return acts, edges


if __name__ == "__main__":
    start = time(); print()
    
    fname = "./results/active_multi.pickle"
    
    # Create noises
    noiseDict = create_noises(sparams, nInstances)
    
    acts, edges = get_activities(fname, df, noiseDict, edges)
    maxVal = edges[np.argmin(np.abs(np.cumsum(acts / acts.sum()) - 0.95))]
    print("%.2f are below %.5f" % (0.95, maxVal))
    
    # Plot activations
    plt.figure(figsize=(3,2))
    plt.bar(edges[:-1], acts / acts.sum(), width=np.diff(edges), color="gray", alpha=0.5)
    
    # Plot Naka-Rushton
    MP, _ = load_params(fname)
    beta, eta, kappa = MP["beta"], MP["eta"], MP["kappa"]
    
    x = np.linspace(0, xmax, 1000)
    plt.plot(x, naka_rushton(x, MP["alpha3"], beta, eta, kappa)[0,:], "--")
    plt.plot(x, naka_rushton(x, MP["alpha2"], beta, eta, kappa)[0,:], "--")
    plt.plot(x, naka_rushton(x, MP["alpha"],  beta, eta, kappa)[0,:], "--")
    plt.axvline(maxVal, color="k", linestyle="--");  plt.xlim(0, xmax)
    plt.yscale("log"); plt.ylim(0.001); plt.yticks([.01, .1, 1],[.01, .1, 1])
    plt.savefig(fname.removesuffix(".pickle") + "_naka.png", dpi=300)
    
    print('Elapsed time: %.2f minutes' % ((time()-start) / 60.))
