"""
This script can be used to compute and plot the empirical and model deviances.
Since we are running the models on different noise instances than they
have been trained on, this takes a while.

@author: Lynn Schmittwilken, Feb 2025
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import psignifit as ps
import pandas as pd
from scipy.stats import norm
from time import time

from functions import  create_noises, load_all_data, reformat_data, load_params, \
    calc_deviance_residual
from plot_psycurves import get_performance

sys.path.insert(1, '../experimental_data')
from exp_params import stim_params as sparams

np.random.seed(0)

nInstances = 30  # number of noise instances used for testing

edge_conds = sparams["edge_widths"]
noise_conds = sparams["noise_types"]
n_edges = len(edge_conds); n_noises = len(noise_conds)


def getHuman():
    vps = ["ls", "mm", "jv", "ga", "sg", "fd"]
    datadir = "../experimental_data/"
    data = load_all_data(datadir, vps)

    # Get pooled / individual experimental data
    dfListPool = []; dfListInd = []
    for ni, n in enumerate(noise_conds):
        for ei, e in enumerate(edge_conds):
            x, nCo, nTr = reformat_data(data, n, e)
            dfListPool.append(
                pd.DataFrame({
                    "noise": [n,]*len(x),
                    "edge": [e,]*len(x),
                    "contrasts": x,
                    "ncorrect": nCo,
                    "ntrials": nTr,
                    })
                )
            for v, vp in enumerate(vps):
                x, nCo, nTr = reformat_data(data[data["vp"]==v], n, e)
                dfListInd.append(
                    pd.DataFrame({
                        "noise": [n,]*len(x),
                        "edge": [e,]*len(x),
                        "contrasts": x,
                        "ncorrect": nCo,
                        "ntrials": nTr,
                        "vp": [v,]*len(x)
                        })
                    )
    dfPool = pd.concat(dfListPool).reset_index(drop=True)
    dfInd = pd.concat(dfListInd).reset_index(drop=True)
    return dfPool, dfInd


def bin_residuals(residuals, bound, bins):
    # Stack residuals
    residuals_stack = np.copy(residuals)
    residuals_stack[residuals_stack < -bound] = -bound
    residuals_stack[residuals_stack > bound] = bound
    
    # Binning
    c, b = np.histogram(residuals_stack, bins)
    
    # Vectors have different length, so repeat first element for our plotting routine
    c = np.append(c[0], c)
    
    # Normalize area under the curve to 1
    c = c / np.trapz(c, dx=b[1]-b[0])
    return c, b


def getModel(rfile1, rfile2, dfPool, dfInd, noiseDict, plotting=False):
    # Set parameters for psignifit
    options = {"sigmoidName": "norm", "expType": "2AFC"}
    colors = ["C2", "C1", "C0"]
    
    best_params1, mparams1 = load_params(rfile1)
    best_params2, mparams2 = load_params(rfile2)
    mparams1["n_trials"], mparams2["n_trials"] = nInstances, nInstances
    vps = np.unique(dfInd["vp"])
    
    # Initialize arrays for deviances per stimulus condition
    devModel1 = np.zeros([n_noises, n_edges])
    devModel2 = np.zeros([n_noises, n_edges])
    devHuman = np.zeros([n_noises, n_edges])
    
    # Prepare and plot results
    if plotting:
        fig, axes = plt.subplots(n_noises, n_edges, figsize=(6, 8), sharey=True, sharex=True)
        fig.subplots_adjust(wspace=0.001, hspace=0.001)
    
    for ni, n in enumerate(noise_conds):
        print(n)
        for ei, e in enumerate(edge_conds[::-1]):
            # Initialize list for deviance residuals per stimulus condition
            devResModel1 = []; devResModel2 = []; devResHuman = []
    
            # Get performance of average observer (all contrasts)
            dfTemp = dfPool[(dfPool["noise"]==n) & (dfPool["edge"]==e)]
            psidata = np.array(
                [dfTemp["contrasts"].to_numpy(),
                  dfTemp["ncorrect"].to_numpy(),
                  dfTemp["ntrials"].to_numpy()]
                )
            res = ps.psignifit(psidata.transpose(), optionsIn=options)
            fit = res['Fit']
            pc_human = (1 - fit[2] - fit[3]) * res['options']['sigmoidHandle'](dfTemp["contrasts"].to_numpy(), fit[0], fit[1]) + fit[3]
            
            # Loop through all contrasts
            for i, c in enumerate(dfTemp["contrasts"].to_numpy()):
                pc_model1 = get_performance(best_params1, mparams1, e, noiseDict[n], c)
                pc_model2 = get_performance(best_params2, mparams2, e, noiseDict[n], c)
    
                # Calculate deviance residuals for each observer and append
                for v, vp in enumerate(vps):
                    dfVP = dfInd[(dfInd["noise"]==n) & (dfInd["edge"]==e) & (dfInd["vp"]==v)]
                    nCorrect = dfVP["ncorrect"].to_numpy()
                    nTrials = dfVP["ntrials"].to_numpy()
    
                    devResModel1.append(calc_deviance_residual(y=nCorrect[i], n=nTrials[i], p=pc_model1))
                    devResModel2.append(calc_deviance_residual(y=nCorrect[i], n=nTrials[i], p=pc_model2))
                    devResHuman.append(calc_deviance_residual(y=nCorrect[i], n=nTrials[i], p=pc_human[i]))
    
            # Calculate deviance per condition (=sum of squared deviance residuals)
            devModel1[ni, ei] = (np.array(devResModel1)**2.).sum()
            devModel2[ni, ei] = (np.array(devResModel2)**2.).sum()
            devHuman[ni, ei] = (np.array(devResHuman)**2.).sum()
            
            # Plot deviance residuals
            if plotting=="model":
                # Stack residuals larger than 5 / smaller than -5 + Binning + Normalizung
                c1, b1 = bin_residuals(devResModel1, 5, 6)
                c2, b2 = bin_residuals(devResModel2, 5, 6)
                c_emp, b_emp = bin_residuals(devResHuman, 5, 6)
        
                # Get x and y for normal distribution + Normalize
                xnorm = np.linspace(-5, 5, 1000)
                ynorm = norm.pdf(xnorm, 0, 1)
                ynorm = ynorm / np.trapz(ynorm, dx=xnorm[1]-xnorm[0])
        
                # Calculate mean residuals (=bias)
                biasModel1 = np.array(devResModel1).mean()
                biasModel2 = np.array(devResModel2).mean()
                
                # Plotting: Model 1 + empirical
                axes[ni, ei].plot(b1, c1, "gray", drawstyle="steps", label="single")
                axes[ni, ei].fill_between(b1, c1, step="pre", alpha=0.8, color="darkgray")
                axes[ni, ei].plot(xnorm, ynorm, "gray")
                axes[ni, ei].plot((biasModel1, biasModel1), (0, -0.9), "gray", linestyle="dashed")
                axes[ni, ei].plot(b_emp, c_emp, colors[ei], drawstyle="steps", label="empirical", lw=2)
        
                # Plotting: Model 2 + empirical
                axes[ni, ei].plot(b2, -c2, "k", drawstyle="steps", label="multi")
                axes[ni, ei].fill_between(b2, -c2, step="pre", alpha=0.5, color="k")
                axes[ni, ei].plot(xnorm, -ynorm, "k")
                axes[ni, ei].plot((biasModel2, biasModel2), (0, 0.9), "k", linestyle="dashed")
                axes[ni, ei].plot(b_emp, -c_emp, colors[ei], drawstyle="steps", lw=2)

            elif plotting=="empirical":
                devResHuman = np.array(devResHuman)
                devResHuman[devResHuman < -3] = -3
                devResHuman[devResHuman >  3] = 3
                c_emp, b_emp = bin_residuals(devResHuman, 5, 6)
                axes[ni, ei].plot(b_emp, c_emp, colors[ei], drawstyle="steps", label="empirical", lw=2)
                axes[ni, ei].fill_between(b_emp, c_emp, step="pre", alpha=0.5, color=colors[ei])
    return devModel1, devModel2, devHuman


if __name__ == "__main__":
    start = time()
    
    # Get human data
    dfPool, dfInd = getHuman()
    
    # Plot pooled curves
    print()
    noiseDict = create_noises(sparams, nInstances)
    
    rf1 = "./results/active_multi.pickle"
    rf2 = "./results/active_spatial.pickle"
    dev1, dev2, devHuman = getModel(rf1, rf2, dfPool, dfInd, noiseDict, plotting="model")
    plt.savefig('./results/devRes_multis.png', dpi=300)
    
    plt.figure(figsize=(4.5, 4))
    plt.subplot(121); plt.imshow(dev1/devHuman, "gray", vmin=1, vmax=8), plt.axis("off")
    plt.subplot(122); plt.imshow(dev2/devHuman, "gray", vmin=1, vmax=8), plt.axis("off")
    plt.savefig('./results/deviance_multis.png', dpi=300)
    
    ndata = 5 * 6
    print("Mean empirical deviance over all conditions is", (devHuman / ndata).mean())
    print("Mean model 1 deviance over all conditions is", (dev1 / ndata).mean())
    print("Mean model 2 deviance over all conditions is", (dev2 / ndata).mean())
    print('Elapsed time: %.2f minutes' % ((time()-start) / 60.))