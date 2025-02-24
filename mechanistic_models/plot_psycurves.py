"""
This script can be used to plot the model psychometric curves alongside
the human data.
Since we are running the models on different noise instances than they
have been trained on, this takes a while.

@author: Lynn Schmittwilken, Feb 2025
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import psignifit as ps
import pickle
from time import time

from functions import create_filter_outputs, compute_performance, create_edge, \
    create_noises, plotPsych, load_all_data, reformat_data, load_params

sys.path.insert(1, '../experimental_data')
from exp_params import stim_params as sparams

np.random.seed(23)

savePickle = True
loadPickle = False

nInstances = 30        # number of noise instances used for testing
n_levels = 30          # number of contrasts for psycurve
xlim = [5e-04, 0.05]   # min and max contrast for psycurve

resultsPath = "./results"

edge_conds = sparams["edge_widths"]
noise_conds = sparams["noise_types"]


def get_performance(params, mparams, e, noiseList, c):
    beta, eta, kappa = params["beta"], params["eta"], params["kappa"]
    alphas = [value for key, value in params.items() if "alpha" in key.lower()]
    
    edge = create_edge(c, e, sparams)
    
    pc = np.zeros(mparams["n_trials"])
    for t in range(mparams["n_trials"]):
        mout1, mout2 = create_filter_outputs(edge, noiseList[t], noiseList[t-1], mparams)
        pc[t] = compute_performance(mout1, mout2, mparams, alphas, beta, eta, kappa)
    return pc.mean()


def get_pcs(params, mparams, e, n, contrast):
    # Append performances
    pcs = np.zeros(len(contrast))
    for ci, c in enumerate(contrast):
        pcs[ci] = get_performance(params, mparams, e, n, c)
    return pcs


def plotHuman(axes):
    vps = ["ls", "mm", "jv", "ga", "sg", "fd"]
    datadir = "../experimental_data/"
    data = load_all_data(datadir, vps)

    # Set parameters for psignifit
    options = {"sigmoidName": "norm", "expType": "2AFC"}
    colors = ["C2", "C1", "C0"]
    cmax = np.zeros([len(edge_conds), len(noise_conds)])
    
    for ni, n in enumerate(noise_conds):
        for ei, e in enumerate(edge_conds[::-1]):
            x, n_correct, n_trials = reformat_data(data, n, e)
            res = ps.psignifit(np.array([x, n_correct, n_trials]).transpose(), optionsIn=options)
            plotPsych(res, color=colors[ei], axisHandle=axes[ni, ei], plotCI=True, xlim=xlim, plotLapse=False)
            cmax[ei,ni] = x.max()
    return cmax


def plotModel(results_file, axes, ltype, color, noiseDict, cmax):
    if loadPickle:
        with open(resultsPath + results_file.split(".")[0] + "_psicurve.pickle", 'rb') as handle:
            data = pickle.load(handle)
            pcs = data["pcs"]
            contrasts = data["contrasts"]
    
    else:
        pcs = np.zeros([len(edge_conds), len(noise_conds), n_levels])
        contrasts = np.zeros(pcs.shape)
        best_params, mparams = load_params(resultsPath + results_file)
        mparams["n_trials"] = nInstances
        for ni, n in enumerate(noise_conds):
            for ei, e in enumerate(edge_conds[::-1]):
                contrasts[ei,ni,:] = np.linspace(xlim[0], cmax[ei,ni]*2., n_levels)
                pcs[ei,ni,:] = get_pcs(best_params, mparams, e, noiseDict[n], contrasts[ei,ni,:])
        
        if savePickle:
            with open(resultsPath + results_file.split(".")[0] + "_psicurve.pickle", 'wb') as handle:
                pickle.dump({"pcs": pcs, "contrasts": contrasts}, handle)
    
    for ni, n in enumerate(noise_conds):
        for ei, e in enumerate(edge_conds[::-1]):
            axes[ni, ei].plot(contrasts[ei,ni,:], pcs[ei,ni,:], ltype, color=color, linewidth=1)


if __name__ == "__main__":
    start = time()
    
    fig, axes = plt.subplots(len(noise_conds), len(edge_conds), figsize=(6, 6), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.001, hspace=0.001)
    
    # Plot human data
    cmax = plotHuman(axes)
    
    # Plot pooled curves
    print()
    noiseDict = create_noises(sparams, nInstances)
    plotModel("/active_multi.pickle", axes, "--", "k", noiseDict, cmax)
    plotModel("/spatial_multi.pickle", axes, "-", "k", noiseDict, cmax)
    
    x = [5e-3, 5e-2]
    axes[3, 0].set(ylabel="Percent correct", xscale="log", xticks=x, xticklabels=x)
    axes[len(noise_conds)-1, 1].set(xlabel="Edge contrast [rms]")
    plt.savefig(resultsPath + '/psicurves_multis.png', dpi=300)
    plt.show()
    
    print('Elapsed time: %.2f minutes' % ((time()-start) / 60.))
