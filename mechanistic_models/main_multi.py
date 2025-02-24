"""
Script to optimize performance of model(s)

@author: Lynn Schmittwilken
Last update: Feb 2025
"""

import numpy as np
import pandas as pd
import pickle
from time import time
import logging
import sys
from scipy.optimize import minimize
from functions import create_loggabors, compute_performance, \
    grid_optimize, log_likelihood, save_filter_outputs, load_filter_outputs, \
    create_drift, create_isologgabors, \
    watson_tf, kelly_csf, zheng_tf, benardete_tf

sys.path.insert(1, '../experimental_data')
from exp_params import stim_params as sparams

np.random.seed(0)


####################################
#           Parameters             #
####################################
results_file = "./results/active_multi.pickle"

# Model params
fos = [0.5, 3., 9.]
sigma_fo = 0.5945                    # from Schütt & Wichmann (2017)
sigma_angleo = 0.2965                # from Schütt & Wichmann (2017)
n_trials = 30                        # average performance over n-trials
noiseVar = 1.                        # magnitude of internal noise

tempBool = True                      # True=active model; False=spatial model
collapseTime = True                  # collapse time dim before normalization (only relevant for active model)
tempType = "watson"                  # which temporal filter? (zheng, watson, kelly, benardete)
Nt = 40; dt = 0.005                  # number of steps; step size (s)
D = 20./(60.**2.)                    # drift diffusion constant

isoSF = False                        # isotropic spatial filter?
gain = None                          # None, global, channel, local, spatial
sameNoise = True                     # use same or different noise instances?
ftype = "pickle"                     # pickle (*), npy, hdf5
outDir = "./outputs"                 # directory to save filter outputs
# outDir = "/dev/shm/datasets/outputs"

n_filters = len(fos)
ppd = sparams["ppd"]
fac = int(ppd*2)                     # padding to avoid border artefacts
sparams["n_masks"] = n_trials        # use same noise masks everytime


####################################
#            Read data             #
####################################
df = pd.read_csv("../experimental_data/expdata_pooled.txt", sep=" ")
noise_conds = np.unique(df["noise"])   # Noise conditions
edge_conds = np.unique(df["edge"])     # Edge conditions


####################################
#           Preparations           #
####################################
# Calculate spatial frequency axis in cpd:
nX = int(sparams["stim_size"] * ppd)
fs = np.fft.fftshift(np.fft.fftfreq(int(nX), d=1./ppd))
fx, fy = np.meshgrid(fs, fs)

# Create spatial filters
if isoSF:
    loggabors = create_isologgabors(fx, fy, fos, sigma_fo)
else:
    loggabors = create_loggabors(fx, fy, fos, sigma_fo, 0., sigma_angleo)

# Create temporal filter(s)
if tempType == "watson":
    tempFilter = watson_tf(Nt, dt)
    tempFilterP = np.fft.fftshift(np.fft.fft(tempFilter)) # fft
elif tempType == "kelly":
    tf = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))
    Tcsf05 = kelly_csf(sfs=[.5,], tfs=tf)
    Tcsf3 = kelly_csf(sfs=[3.,], tfs=tf)
    Tcsf9 = kelly_csf(sfs=[9.,], tfs=tf)
    tempFilterP = np.array([Tcsf05, Tcsf3, Tcsf9])
elif tempType == "zheng":
    tempFilterP = zheng_tf(Nt, dt)
elif tempType == "benardete":
    tempFilterP = benardete_tf(Nt, dt)

tempFilterP = tempFilterP / tempFilterP.max()  # Normalize for now


# Create drift traces
drift = []
for t in range(n_trials):
    _, driftInt = create_drift(Nt*dt-dt, 1./dt, ppd, D)
    drift.append(driftInt)

# Constant model params
mparams = {"n_filters": n_filters,
           "fos": fos,
           "sigma_fo": sigma_fo,
           "sigma_angleo": sigma_angleo,
           "loggabors": loggabors,
           "fac": fac,
           "nX": nX,
           "n_trials": n_trials,
           "gain": gain,
           "outDir": outDir,
           "sameNoise": sameNoise,
           "noiseVar": noiseVar,
           "Nt": Nt,
           "dt": dt,
           "D": D,
           "drift": drift,
           "tempFilterP": tempFilterP,
           "tempBool": tempBool,
           "collapseTime": collapseTime,
           "tempType": tempType,
           "isoSF": isoSF,
           }

adict = {"model_params": mparams, "stim_params": sparams}

def get_loss(params):
    # Read params from dict / list
    if type(params) is dict:
        beta, eta, kappa = params["beta"], params["eta"], params["kappa"]
        alphas = [value for key, value in params.items() if "alpha" in key.lower()]

    else:
        beta, eta, kappa = params[0], params[1], params[2]
        alphas = params[3::]

    # Infinite loss if all alphas are zero
    if sum(alphas) == 0:
        return np.inf
    
    # Infinite loss if alphas are negative
    if any(x < 0 for x in alphas):
        return np.inf
    if any(x < 0 for x in [beta, eta, kappa]):
        return np.inf
    if beta == 0:
        beta = 1e-18  # avoid division by zero in Naka-Rushton

    # Run model for each contrast in each condition
    LLs = []
    for n in noise_conds:
        for e in edge_conds:
            df_cond = df[(df["noise"]==n) & (df["edge"]==e)]
            contrasts = df_cond["contrasts"].to_numpy()
            ncorrect = df_cond["ncorrect"].to_numpy()
            ntrials = df_cond["ntrials"].to_numpy()
            lamb = np.unique(df_cond["lambda"].to_numpy())[0]

            for i, c in enumerate(contrasts):
                pc = np.zeros(n_trials)
                
                for t in range(n_trials):
                    mparams["trial"] = t
                    
                    # Load filter outputs
                    name = outDir + "/%s_%.3f_%i_%i" % (n, e, i, t)
                    mout1, mout2 = load_filter_outputs(name, ftype)
                    pc[t] = compute_performance(mout1, mout2, mparams, alphas, beta, eta, kappa, lamb)

                # Compute log-likelihood
                LLs.append(log_likelihood(y=ncorrect[i], n=ntrials[i], p=pc.mean()))
    #print(params, -sum(LLs))
    return -sum(LLs)


####################################
#           Optimization           #
####################################
if __name__ == "__main__":
    start = time()
    
    # Set up log-file
    logname = "log_" + results_file.rsplit("/")[-1].rsplit(".")[0] + ".txt"
    logging.basicConfig(filename=logname, format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.info('Logging app started')
    
    # Filter outputs dont change with model params, so save+load to reduce runtime drastically
    print('------------------------------------------------')
    print('Creating all filter outputs and save them to memory for computational efficiency')
    logging.info('Dataset generation started')
    save_filter_outputs(sparams, mparams, df, outDir, ftype)
    logging.info('Dataset generation finished')
    
    # Initial parameter guesses for grid search
    alpha = np.linspace(0., 5, 6)
    params_dict = {
        "beta": [1e-15, 1e-10, 1e-5, 1e-0],
        "eta": np.linspace(0., 1., 6),
        "kappa": np.linspace(0., 10., 6),
        "alpha": alpha,
        "alpha2": alpha,
        "alpha3": alpha,
        }
    
    # Run / Continue optimization
    print('------------------------------------------------')
    print('Starting optimization:', results_file)
    print('------------------------------------------------')
    logging.info('Grid search started')
    bparams, bloss = grid_optimize(results_file, params_dict, get_loss, adict)
    logging.info('Grid search finished')
    logging.info(f'Best loss {bloss}')
    logging.info(f'Best params {bparams}')
    
    print()
    print("Best params:", bparams)
    print("Best loss:", bloss)
    print('------------------------------------------------')
    print('Elapsed time: %.2f minutes' % ((time()-start) / 60.))
    
    # Use best params from grid search for initial guess for optimizer
    automatic_grid = True
    if automatic_grid:
        logging.info('Automatic optimization started')
        start = time()
        res = minimize(get_loss,
                       list(bparams.values()),
                       method='Nelder-Mead',      # Nelder-Mead (=Simplex)
                       options={"maxiter": 500},
                       )
        logging.info('Automatic optimization finished')
        logging.info(f'Best loss {res.fun}')
        logging.info(f'Best params {res.x}')
        
        # Save final results to pickle
        with open(results_file, 'rb') as handle:
            data_pickle = pickle.load(handle)
    
        xf = 1
        data_pickle["best_loss_auto"] = res.fun
        data_pickle["best_params_auto"] = {
            "beta": res.x[0],
            "eta": res.x[0+xf],
            "kappa": res.x[1+xf],
            "alpha": res.x[2+xf],
            "alpha2": res.x[3+xf],
            "alpha3": res.x[4+xf],
            }
        data_pickle["overview_auto"] = res
            
        with open(results_file, 'wb') as handle:
            pickle.dump(data_pickle, handle)
    
        print('------------------------------------------------')
        print(res)
        print('------------------------------------------------')
        print('Elapsed time: %.2f minutes' % ((time()-start) / 60.))
