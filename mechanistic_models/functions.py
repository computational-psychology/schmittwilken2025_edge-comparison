"""
These functions are the basis for the modeling part

@author: Lynn Schmittwilken
Last update: Feb 2025
"""

import numpy as np
import numpy.lib.format
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm, binom
from scipy.signal import fftconvolve
import os
import pickle
import itertools
import math
from stimupy.stimuli.cornsweets import cornsweet_edge
from stimupy.noises.whites import white as create_whitenoise
from stimupy.noises.narrowbands import narrowband as create_narrownoise
from stimupy.noises.naturals import one_over_f as create_pinknoise

sys.path.insert(1, '../')
import psignifit as ps

# %%
###############################
#       Helper functions      #
###############################
def print_progress(count, total):
    """Helper function to print progress.

    Parameters
    ----------
    count
        Current iteration count.
    total
        Total number of iterations.

    """
    percent_complete = float(count) / float(total)
    msg = "\rProgress: {0:.1%}".format(percent_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def create_directory(dir_path, askPermission=False):
    if os.path.exists(dir_path) == 1:
        if askPermission:
           out = input("Directory already exists. Override content? (y / n)")
           if not (out == "y" or out == "yes"): raise SystemExit(0)
    else:
        os.makedirs(dir_path)


def remove_padding(array, rb, axis=1):
    array_shape = array.shape
    
    if len(array_shape) == 2:
        if axis==0:   array = array[rb:array_shape[0]-rb, :]
        elif axis==1: array = array[:, rb:array_shape[1]-rb]
    elif len(array_shape) == 3:
        if axis==0:   array = array[rb:array_shape[0]-rb, :, :]
        elif axis==1: array = array[:, rb:array_shape[1]-rb, :]
        elif axis==2: array = array[:, :, rb:array_shape[2]-rb]
    return array


def add_padding(array, rb, val, axis=1):
    array_shape = array.shape
    if len(array_shape) == 2:
        if axis==0:
            new_array = np.ones([array_shape[0]+rb*2, array_shape[1]]) * val
            new_array[rb:array_shape[0]+rb, :] = array
        elif axis==1:
            new_array = np.ones([array_shape[0], array_shape[1]+rb*2]) * val
            new_array[:, rb:array_shape[1]+rb] = array
    elif len(array_shape) == 3:
        if axis==0:
            new_array = np.ones([array_shape[0]+rb*2, array_shape[1], array_shape[2]]) * val
            new_array[rb:array_shape[0]+rb, :, :] = array
        elif axis==1:
            new_array = np.ones([array_shape[0], array_shape[1]+rb*2, array_shape[2]]) * val
            new_array[:, rb:array_shape[1]+rb, :] = array
        elif axis==2:
            new_array = np.ones([array_shape[0], array_shape[1], array_shape[2]+rb*2]) * val
            new_array[:, :, rb:array_shape[2]+rb] = array
    return new_array


def load_params(results_file):
    # Load data from pickle:
    with open(results_file, 'rb') as handle:
        data_pickle = pickle.load(handle)
    
    best_params = data_pickle["best_params_auto"]
    best_loss = data_pickle["best_loss_auto"]
    model_params = data_pickle["model_params"]
    print("Best loaded loss:", best_loss)
    return best_params, model_params


def getConj(v1, v2):
    return np.fft.fftshift(np.append(v1[:-1], np.conj(np.flip(v2[1::]))))


# %%
###############################
#       Stimulus-related      #
###############################
def create_edge(contrast, width, stim_params):
    e = cornsweet_edge(
        visual_size=stim_params["stim_size"],
        ppd=stim_params["ppd"],
        ramp_width=width,
        exponent=stim_params["edge_exponent"],
        intensity_edges=[-1, 1],
        intensity_plateau=0,
        )["img"]
    e = contrast * stim_params["mean_lum"] * e/e.std() + stim_params["mean_lum"]
    return e


def load_mask(file_name):
    # Load data from pickle:
    with open(file_name, 'rb') as handle:
        data_pickle = pickle.load(handle)
    return data_pickle["noise"], data_pickle["stimulus_params"]


def create_noise(n, sparams):
    sp = sparams
    ssize = sp["stim_size"]
    ppd = sp["ppd"]
    rms = sp["noise_contrast"] * sp["mean_lum"]
    
    if n == "none":
        noise = np.zeros([int(ssize*ppd), int(ssize*ppd)])
    elif n == "white":
        noise = create_whitenoise(visual_size=ssize, ppd=ppd, pseudo_noise=True)["img"]
    elif n == "pink1":
        noise = create_pinknoise(visual_size=ssize, ppd=ppd, exponent=1., pseudo_noise=True)["img"]
    elif n == "pink2":
        noise = create_pinknoise(visual_size=ssize, ppd=ppd, exponent=2., pseudo_noise=True)["img"]
    elif n == "narrow0.5":
        noise = create_narrownoise(visual_size=ssize, ppd=ppd, center_frequency=0.5, bandwidth=1., pseudo_noise=True)["img"]
    elif n == "narrow3":
        noise = create_narrownoise(visual_size=ssize, ppd=ppd, center_frequency=3, bandwidth=1., pseudo_noise=True)["img"]
    elif n == "narrow9":
        noise = create_narrownoise(visual_size=ssize, ppd=ppd, center_frequency=9, bandwidth=1., pseudo_noise=True)["img"]

    if not n=="none":
        noise = noise - noise.mean()
        noise = noise / noise.std() * rms
    return noise


def create_noises(sparams, nInstances):
    noiseDict = {}
    for ni, n in enumerate(sparams["noise_types"]):
        noiseList = []
        for t in range(nInstances):
            noiseList.append(create_noise(n, sparams))
        noiseDict[n] = noiseList
    return noiseDict


def pull_noise_mask(noise_type, sparams, trial, mask_path="./noise_masks/"):
    # maskID = np.random.randint(0, sparams["n_masks"])
    maskID = trial
    if noise_type == 'none':
        nX = int(sparams["stim_size"]*sparams["ppd"])
        noise = np.zeros([nX, nX])
    elif noise_type == 'white':
        noise, _ = load_mask(mask_path + "white/" + str(maskID) + ".pickle")
    elif noise_type == 'pink1':
        noise, _ = load_mask(mask_path + "pink1/" + str(maskID) + ".pickle")
    elif noise_type == 'pink2':
        noise, _ = load_mask(mask_path + "pink2/" + str(maskID) + ".pickle")
    elif noise_type == 'narrow0.5':
        noise, _ = load_mask(mask_path + "narrow0.5/" + str(maskID) + ".pickle")
    elif noise_type == 'narrow3':
        noise, _ = load_mask(mask_path + "narrow3/" + str(maskID) + ".pickle")
    elif noise_type == 'narrow9':
        noise, _ = load_mask(mask_path + "narrow9/" + str(maskID) + ".pickle")
    else:
        raise ValueError("noise_type unknown")
    return noise


# %%
###############################
#           Filters           #
###############################
def create_lowpass(fx, fy, radius, sharpness):
    # Calculate the distance of each frequency from requested frequency
    distance = radius - np.sqrt(fx**2. + fy**2.)
    distance[distance > 0] = 0
    distance = np.abs(distance)

    # Create bandpass filter:
    lowpass = 1. / (np.sqrt(2.*np.pi) * sharpness) * np.exp(-(distance**2.) / (2.*sharpness**2.))
    lowpass = lowpass / lowpass.max()
    return lowpass


def create_loggabor_fft(fx, fy, fo, sigma_fo, angleo, sigma_angleo):
    nY, nX = fx.shape
    fr = np.sqrt(fx**2. + fy**2.)
    fr[int(nY/2), int(nX/2)] = 1.

    # Calculate radial component of the filter:
    radial = np.exp((-(np.log(fr/fo))**2.) / (2. * np.log(sigma_fo)**2.))
    radial[int(nY/2), int(nX/2)] = 0.  # Undo radius fudge
    fr[int(nY/2), int(nX/2)] = 0.      # Undo radius fudge

    # Multiply radial part with lowpass filter to achieve even coverage in corners
    # Lowpass filter will limit the maximum frequency
    lowpass = create_lowpass(fx, fy, radius=fx.max(), sharpness=1.)
    radial = radial * lowpass

    # Calculate angular component of log-Gabor filter
    theta = np.arctan2(fy, fx)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # For each point in the polar-angle-matrix, calculate the angular distance
    # from the filter orientation
    ds = sintheta * np.cos(angleo) - costheta * np.sin(angleo)  # difference in sin
    dc = costheta * np.cos(angleo) + sintheta * np.sin(angleo)  # difference in cos
    dtheta = np.abs(np.arctan2(ds, dc))                         # absolute angular distance
    angular = np.exp((-dtheta**2.) / (2. * sigma_angleo**2.))   # calculate angular filter component
    return angular * radial  # loggabor is multiplication of both


def create_loggabor(fx, fy, fo, sigma_fo, angleo, sigma_angleo):
    # Create loggabor and ifft
    loggabor_fft = create_loggabor_fft(fx, fy, fo, sigma_fo, angleo, sigma_angleo)
    loggabor = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(loggabor_fft)))
    loggabor_even = loggabor.real          # real part = even-symmetric filter
    loggabor_odd = np.real(loggabor.imag)  # imag part = odd-symmetric filter
    return loggabor_even, loggabor_odd


def create_loggabors(fx, fy, fos, sigma_fo, angleo, sigma_angleo):
    n_filters = len(fos)
    loggabors = []
    for f in range(n_filters):
        _, loggabor_odd = create_loggabor(fx, fy, fos[f], sigma_fo, 0., sigma_angleo)
        loggabors.append(loggabor_odd)
    return loggabors


def create_isologgabors(fx, fy, fos, sigma_fo):
    n_filters = len(fos)
    loggabors = []
    for f in range(n_filters):
        loggabor, _ = create_loggabor(fx, fy, fos[f], sigma_fo, 0., np.inf)
        loggabors.append(loggabor)
    return loggabors


def create_loggabors_fft(fx, fy, fos, sigma_fo, angleo, sigma_angleo):
    n_filters = len(fos)
    loggabors = []
    for f in range(n_filters):
        loggabor = create_loggabor_fft(fx, fy, fos[f], sigma_fo, 0., sigma_angleo)
        loggabors.append(loggabor)
    return loggabors


def create_gauss_fft(fx, fy, sigma: float):
    gauss = np.exp(-2. * np.pi**2. * sigma**2. * (fx**2. + fy**2.))
    return gauss


def kelly_csf(sfs, tfs):
    Sfs = np.abs(sfs); Tfs = np.abs(tfs) # for negative sfs/tfs
    idx=np.where(Sfs==0.); Sfs[idx]=1.   # fudge for sf=0
    idx2=np.where(Tfs==0.); Tfs[idx2]=1. # fudge for tf=0
    v = Tfs / Sfs                        # calculate "velocity"
    
    # Calculate contrast sensitivity function:
    k = 6.1 + 7.3 * np.abs(np.log10(v/3.))**3.
    amax = 45.9 / (v + 2.)
    csf = k * v * (2.*np.pi*Sfs)**2. * np.exp((-4.*np.pi*Sfs) / amax)
    
    if len(idx):
        csf[idx]=0.; Sfs[idx]=0.   # undo fudge
    if len(idx2):
        csf[idx2]=0.; Tfs[idx2]=0. # undo fudge
    return csf

def watson_tf(Nt=100, dt=0.001):
    # Watson1986-parameters to fit Robson1966 data. Zeta is .9 but then DC!=0
    kappa = 1.33; n1 = 9; n2 = 10; tau = 4.3/1000; zeta = 1; ksi = 269

    # Compute impulse response
    t = np.arange(0, Nt, 1) * dt
    h1 = (t / tau)**(n1 - 1) * np.exp(-t / tau) / (tau * math.factorial(n1-1))
    h2 = (t / (tau*kappa))**(n2 - 1) * np.exp(-t / (tau*kappa)) / (tau*kappa * math.factorial(n2-1))
    h = ksi * (h1 - zeta * h2) * dt
    return h / h.max()


def zheng_tf(Nt=100, dt=0.001):
    m1 = 69.3; m2 = 22.9; m3 = 8.1; m4 = 0.8  # Zheng2007-params
    
    # Compute transfer function
    w = np.linspace(0, .5*(1/dt), int(Nt/2)+1); w[0] = 1. # prevent divison by 0
    H = m1 * np.exp(-(w / m2) ** 2.) / (1. + (m3 / w)**m4)
    
    # Add phase information
    x = np.linspace(0., .5*(1/dt), len(H))
    H = H * np.sin(x) + H * np.cos(x) * 1j
    
    # Perform ifft to get impulse response
    w[0] = 0; w = getConj(w, -w); H[0] = H[0]*0; H = getConj(H, H)
    # h = np.real(np.fft.ifft(np.fft.ifftshift(H)))
    return np.abs(H)


def benardete_tf(Nt=100, dt=0.001):
    # Benardete1999-params. Units in seconds. Hs is .98 but then integral of impulse function != 0
    c = .4; A = 567; D = 2.2/1000.; C_12 = 0.056; T0 = 54.6; tau_L = 1.41/1000.; N_L = 30.3; Hs = 1.

    # Compute transfer function
    w = np.linspace(0, .5*(1./dt), int(Nt/2)+1); w2pij = w*2.*np.pi*1j
    H = A * np.exp(-w2pij*D) * (1. - Hs/(1. + w2pij*(T0/(1.+(c/C_12)**2.))/1000.)) * ((1./(1.+w2pij*tau_L))**N_L)
    H = getConj(H, H)
    # h = np.real(ufft.ifft(ufft.ifftshift(H))) # Perform ifft to get impulse response
    return np.abs(H)
    


# %%
###############################
#            Drift            #
###############################
def brownian(T: float, pps: float, D: float):
    n = int(T*pps)  # Number of drift movements
    dt = 1. / pps   # Time step between two consequent steps (unit: seconds)

    # Generate a 2d stochastic, normally-distributed time series:
    y = np.random.normal(0, 1., [2, n])

    # The average displacement is proportional to dt and D
    y = y * np.sqrt(2.*dt*D)

    # Set initial displacement to 0.
    y = np.insert(y, 0, 0., axis=1)
    return y


def create_drift(T, pps, ppd, D):
    # Our simulations are in px-space. Let's ensure that drift paths != 0
    cond = 0.
    while (cond == 0.):
        # Generate 2d brownian displacement array
        y = brownian(T, pps, D) * ppd

        # Generate drift path in px from continuous displacement array
        y = np.cumsum(y, axis=-1)
        y_int = np.round(y).astype(int)

        # Sum horizontal and vertical drifts in px. Make sure that they are !=0
        cond = y_int[0, :].sum() * y_int[1, :].sum()
    return y, y_int


def apply_drift(stimulus, drift, back_lum=0):
    steps = np.size(drift, 1)
    center_x1 = int(np.size(stimulus, 0) / 2)
    center_y1 = int(np.size(stimulus, 1) / 2)

    # Determine the largest displacement and increase stimulus size accordingly
    largest_disp = int(np.abs(drift).max())
    stimulus_extended = np.pad(stimulus, largest_disp, 'constant', constant_values=(back_lum))
    center_x2 = int(np.size(stimulus_extended, 0) / 2)
    center_y2 = int(np.size(stimulus_extended, 1) / 2)

    # Initialize drift video:
    stimulus_video = np.zeros([np.size(stimulus, 0), np.size(stimulus, 1), steps], np.float16)
    stimulus_video[:, :, 0] = stimulus

    for t in range(1, steps):
        x, y = int(drift[0, t]), int(drift[1, t])

        # Create drift video:
        stimulus_video[:, :, t] = stimulus_extended[
                center_x2-center_x1+x:center_x2+center_x1+x,
                center_y2-center_y1+y:center_y2+center_y1+y]
    return stimulus_video.astype(np.float32)


# %%
###############################
#            Model            #
###############################
# Naka Rushton adapted from Felix thesis
def naka_rushton(inpt, alpha, beta, eta, kappa, gain=None):
    if len(inpt.shape) < 4:
        alpha = np.expand_dims(np.array(alpha), axis=(0, 1))
    else:
        alpha = np.expand_dims(np.array(alpha), axis=(0, 1, -1))
    denom = inpt**kappa
    
    # Additional gain control mechanisms
    if gain == "global": # norm by global activity
        denom = denom.mean()
    elif gain == "channel": # norm by global activity within channels
        denom = denom / denom.mean((0, 1))
    elif gain == "local": # norm by local activity across channels
        denom = denom / np.expand_dims(denom.mean(2), -1)
    elif gain == "spatial": # norm by neighboring activity within channels
        lgsize = [94, 16, 4]
        for i in range(len(lgsize)):
            convf = np.ones([lgsize[i], lgsize[i]]) / lgsize[i]**2.
            denom[:,:,i] = fftconvolve(denom[:,:,i], convf, 'same')
    return alpha * inpt**(eta+kappa) / (denom + beta)


def apply_filters(stim, mparams):
    stim = add_padding(stim, mparams["fac"], stim.mean(), axis=1) # padding
    if mparams["isoSF"]:
        # for the odd-symmetric filter, we did not need padding in filter orientation
        stim = add_padding(stim, mparams["fac"], stim.mean(), axis=0)
    
    # further remove border artefacts through masking
    fc = int(mparams["nX"] * 0.15)
    fc += fc % 2
    mask = np.pad(np.ones([mparams["nX"]-fc, mparams["nX"]-fc]), (int(fc/2), int(fc/2)))

    if mparams["tempBool"] and not mparams["collapseTime"]:
        out = np.zeros([mparams["nX"], mparams["nX"], mparams["n_filters"], mparams["Nt"]])
        mask = np.expand_dims(mask, (-1,-2))
    else:
        out = np.zeros([mparams["nX"], mparams["nX"], mparams["n_filters"]])
        mask = np.expand_dims(mask, -1)
    
    # Spatial filtering
    for fil in range(mparams["n_filters"]):
        outTemp = fftconvolve(stim, mparams["loggabors"][fil], mode='same')
        outTemp = remove_padding(outTemp, mparams["fac"], axis=1)
        if mparams["isoSF"]:
            outTemp = remove_padding(outTemp, mparams["fac"], axis=0)
    
        # Temporal filtering (padding in time does not change output)
        if mparams["tempBool"]:
            # temporal filtering in freq space because it seems more robust
            outTemp = apply_drift(outTemp, mparams["drift"][mparams["trial"]], outTemp.mean())
            if mparams["tempType"] == "kelly":
                thisFilt = np.expand_dims(mparams["tempFilterP"][fil,:], (0,1))
            else:
                thisFilt = np.expand_dims(mparams["tempFilterP"], (0,1))
            outTemp = np.fft.fftshift(np.fft.fftn(outTemp)) * thisFilt
            outTemp = np.abs(np.real(np.fft.ifftn(np.fft.ifftshift(outTemp))))
            if mparams["collapseTime"]:
                out[:, :, fil] = outTemp.mean(2) # collapse temporal dim
            else:
                out[:, :, fil, :] = outTemp
        else:
            out[:, :, fil] = np.abs(outTemp)
    return out * mask


def create_filter_outputs(edge, noise1, noise2, mparams):
    mout1 = apply_filters(edge+noise1, mparams)
    
    if mparams["sameNoise"]:
        mout2 = apply_filters(noise1+edge.mean(), mparams)
    else:
        mout2 = apply_filters(noise2+edge.mean(), mparams)
        
        # if different noises are used, weight by edge profile
        sweight = np.abs(edge - edge.mean())
        mout1 = mout1 * np.expand_dims(sweight/sweight.max(), -1)
        mout2 = mout2 * np.expand_dims(sweight/sweight.max(), -1)
    return mout1, mout2


def compute_dprime(r1, r2, lamb=0.005, noiseVar=1.):
    dprime = (r1 - r2).sum() / np.sqrt( noiseVar*np.ones(r1.shape).sum())
    pc = norm.cdf(dprime)
    return lamb + (1. - 2.*lamb) * pc  # consider lapses


def compute_performance(mout1, mout2, mparams, alphas, beta, eta, kappa, lamb=0.005):
    # Apply Naka-rushton to filter outputs
    mout1 = naka_rushton(mout1, alphas, beta, eta, kappa, mparams["gain"])
    mout2 = naka_rushton(mout2, alphas, beta, eta, kappa, mparams["gain"])
    pc = compute_dprime(mout1, mout2, lamb, mparams["noiseVar"])  # dprime
    return pc


def save_filter_outputs(sparams, mparams, df, outDir, ftype="pickle", askPermission=True):
    create_directory(outDir, askPermission)   # Create directory
    
    for n in sparams["noise_types"]:
        for e in sparams["edge_widths"]:
            df_cond = df[(df["noise"]==n) & (df["edge"]==e)]
            contrasts = df_cond["contrasts"].to_numpy()
            for i, c in enumerate(contrasts):
                for t in range(mparams["n_trials"]):
                    # Create stims
                    mparams["trial"] = t
                    noise1 = pull_noise_mask(n, sparams, t)
                    noise2 = pull_noise_mask(n, sparams, (t+1)%mparams["n_trials"])
                    edge = create_edge(c, e, sparams)
                    mout1, mout2 = create_filter_outputs(edge, noise1, noise2, mparams)
                    
                    results_file = outDir + "/%s_%.3f_%i_%i" % (n, e, i, t)
                    print(results_file)
                    if ftype == "pickle":
                        with open(results_file + ".pickle", 'wb') as handle:
                            save_dict = {"mout1": mout1.astype(np.float16),
                                         "mout2": mout2.astype(np.float16),
                                         }
                            pickle.dump(save_dict, handle)

                    elif ftype == "npy":
                        data = np.append(np.expand_dims(mout1, 0),
                                         np.expand_dims(mout2, 0),
                                         axis=0).astype(np.float16)
                        np.save(results_file, data)


def load(file):
    # Fast numpy load: https://github.com/divideconcept/fastnumpyio
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))


def load_filter_outputs(inDir, ftype="pickle"):
    if ftype == "pickle":
        # s = time()
        with open(inDir + ".pickle", 'rb') as handle:
            data = pickle.load(handle)
            mout1, mout2 = data["mout1"], data['mout2']
        # print(time()-s)

    elif ftype == "npy":
        # s = time()
        data = load(inDir + ".npy")
        mout1, mout2 = data[0], data[1]
        # print(time()-s)
    return mout1.astype(np.float32), mout2.astype(np.float32)


# %%
###############################
#         Read data           #
###############################
def load_data(datadir):
    all_data = []

    # Read experimental data from all the subfolders
    filenames = os.listdir(datadir)
    for filename in filenames:
        if filename.startswith("experiment_"):
            data = pd.read_csv(os.path.join(datadir, filename), delimiter='\t')
            all_data.append(data)
    all_data = pd.concat(all_data) # Concat dataframes

    # Reset the indices
    all_data = all_data.reset_index(drop=True)
    return all_data


def load_data_two_sessions(datadir1, datadir2):
    df1 = load_data(datadir1)
    df1['session'] = 0
    try:
        df2 = load_data(datadir2)
        df2['session'] = 1
        df = pd.concat([df1, df2])
    except:
        df = df1
    df = df.reset_index(drop=True)
    return df


def load_all_data(folder_dir, subject_dirs):
    n_sub = len(subject_dirs)
    data_all = []

    for i in range(n_sub):
        datadir1 = folder_dir + subject_dirs[i] + '/experiment'
        datadir2 = folder_dir + subject_dirs[i] + '2/experiment'
        data_sub = load_data_two_sessions(datadir1, datadir2)

        # Add a column for the subject ID
        vp_id = np.repeat(i, data_sub.shape[0])
        data_sub['vp'] = vp_id

        # Add data of individuals to the full dataset
        data_all.append(data_sub)

    # Concat dataframes of all subjects:
    data_all = pd.concat(data_all)
    return data_all


def reformat_data(data, noise_cond, edge_cond):
    # Get data from one condition
    data = data[(data["noise"] == noise_cond) &
                (data["edge_width"] == edge_cond)]
    
    # For each contrast, get number of (correct) trials
    contrasts = np.unique(data["edge_contrast"])
    n_correct = np.zeros(len(contrasts))
    n_trials = np.zeros(len(contrasts))
    for i in range(len(contrasts)):
        data_correct = data[data["edge_contrast"] == contrasts[i]]["correct"]
        n_correct[i] = np.sum(data_correct)
        n_trials[i] = len(data_correct)
    return contrasts, n_correct, n_trials


def get_lapse_rate(contrasts, ncorrect, ntrials):
    psignifit_data = np.array([contrasts, ncorrect, ntrials]).transpose()
    
    options = {
        "sigmoidName": "norm",
        "expType": "2AFC",
        }

    sys.stdout = open(os.devnull, 'w')                # suppress print
    results = ps.psignifit(psignifit_data, options)
    lamb = results["Fit"][2]
    sys.stdout = sys.__stdout__                       # enable print
    return lamb


# %%
###############################
#         Optimization        #
###############################
# Function which computes log-likelihood
def log_likelihood(y, n, p):
    # y: hits, n: trials, p: model performance
    return binom.logpmf(y, n, p).sum()


def calc_deviance_residual(y, n, p, eta=1e-08):
    # Calculate deviance residuals as explained in Wichmann & Hill (2001)
    # y: hits, n: trials, p: model performance
    phuman = y/n
    p1 = n * phuman * np.log(eta + phuman/p)
    p2 = n * (1 - phuman) * np.log( eta + (1-phuman) / (1-p+eta) )
    out = np.sign(phuman-p) * np.sqrt(2*(p1 + p2))
    return out


def grid_optimize(results_file, params_dict, loss_func, additional_dict=None):
    # Check whether pickle exists. If so, load data from pickle file and continue optimization
    if os.path.isfile(results_file):
        print("-----------------------------------------------------------")
        print("Results file %s already exists" % results_file)
        print("Loading optimization data from pickle...")
        print("-----------------------------------------------------------")

        # Load data from pickle:
        with open(results_file, 'rb') as handle:
            data_pickle = pickle.load(handle)

        params_dict = data_pickle["params_dict"]
        best_loss = data_pickle['best_loss']
        best_params = data_pickle['best_params']
        last_set = data_pickle['last_set']

    else:
        print("-----------------------------------------------------------")
        print("Results file %s not found" % results_file)
        print("Initiating optimization ...")
        print("-----------------------------------------------------------")
        best_loss = np.inf
        best_params = []
        last_set = 0

    # Prepare list of params to run:
    n_params = len(params_dict.keys())
    print("Specified %i parameters." % n_params)
    print(params_dict.keys())
    print("-----------------------------------------------------------")
    print("Best starting loss", best_loss)
    print("Best starting params", best_params)
    print("-----------------------------------------------------------")

    # Get all individual parameter lists
    pind = []
    for key in params_dict.keys():
        pind.append(params_dict[key])

    # Get list with all parameter combinations
    pall = list(itertools.product(*pind))
    total = len(pall)

    for i in range(last_set, total):
        # Print progress and get relevant params:
        print_progress(count=i+1, total=total)
        p = {}
        for j, key in enumerate(params_dict.keys()):
            p[key] = pall[i][j]
        loss = loss_func(p)

        if loss < best_loss:
            best_loss = loss
            best_params = p
            print()
            print('New best loss:', best_loss)
            print('Parameters: ', best_params)
            print("-----------------------------------------------------------")

        # Save params and results in pickle:
        save_dict = {'params_dict': params_dict,
                     'last_set': i+1,
                     'best_loss': best_loss,
                     'best_params': best_params,
                     'total': total,
                     }

        if additional_dict is not None:
            save_dict.update(**additional_dict)

        with open(results_file, 'wb') as handle:
            pickle.dump(save_dict, handle)
    return best_params, best_loss


def varyParams(results_file, params_dict, loss_func, best_params, additional_dict=None):
    # Get all parameter combinations
    pall = [list(best_params.values()), ]
    for ki, key in enumerate(list(best_params)):
        pind = list(best_params.values())         # basis are best params
        pind = [[p,] for p in pind]               # each param as list for itertools
        pind[ki] = params_dict[key]               # replace individual param by selection
        pall += list(itertools.product(*pind))    # create all combinations and concat
    total = len(pall)

    # Check whether pickle exists. If so, load data from pickle file and continue optimization
    if os.path.isfile(results_file):
        print("-----------------------------------------------------------")
        print("Results file %s already exists" % results_file)
        print("Loading variation data from pickle...")
        print("-----------------------------------------------------------")

        # Load data from pickle:
        with open(results_file, 'rb') as handle:
            data_pickle = pickle.load(handle)

        params_dict = data_pickle['params_dict']
        best_params = data_pickle['best_params']
        last_set = data_pickle['last_set']
        losses = data_pickle['losses']

    else:
        print("-----------------------------------------------------------")
        print("Results file %s not found" % results_file)
        print("Initiating parameter variation ...")
        print("-----------------------------------------------------------")
        last_set = 0
        losses = np.zeros(total)
    
    # Initiate datarame with params
    df = pd.DataFrame(pall)
    df.columns = list(best_params)
    
    for i in range(last_set, total):
        # Print progress and get relevant params:
        print_progress(count=i+1, total=total)
        p = {}
        for j, key in enumerate(params_dict.keys()):
            p[key] = pall[i][j]
        losses[i] = loss_func(p)
        df["losses"] = losses      # update losses in dataframe

        # Save params and results in pickle:
        save_dict = {'params_dict': params_dict,
                     'last_set': i+1,
                     'best_params': best_params,
                     'total': total,
                     'df': df,
                     'losses': losses,
                     }
        
        if additional_dict is not None:
            save_dict.update(**additional_dict)
        
        with open(results_file, 'wb') as handle:
            pickle.dump(save_dict, handle)
    return df


# %%
###############################
#        Visualizations       #
###############################
def plotPsych(result,
              color          = [0, 0, 0],
              alpha          = 1,
              plotData       = True,
              lineWidth      = 1,
              plotAsymptote  = True,
              extrapolLength = .2,
              dataSize       = 0,
              axisHandle     = None,
              showImediate   = False,
              plotCI         = False,
              xlim           = None,
              plotLapse      = True,
              ):
    """
    This function plots the fitted psychometric function alongside 
    the data. Adapted from the Python-Psignifit plotPsych-function

    Author: Lynn Schmittwilken
    """
    
    fit, data, options = result['Fit'], result['data'], result['options']
    if np.isnan(fit[3]): fit[3] = fit[2]
    if data.size == 0: return
    if dataSize == 0: dataSize = 10000. / np.sum(data[:,2])
    ax = plt.gca() if axisHandle == None else axisHandle
    ax.grid(True, "minor", color=[0.9,]*3); ax.minorticks_on(); ax.grid(True, color=[0.8,]*3)
    lapse = fit[2] if plotLapse else 0.
    
    # PLOT DATA
    ymin = 1. / options['expN']
    ymin = min([ymin, min(data[:,1] / data[:,2])])
    xData = data[:,0]
    if plotData:
        yData = data[:,1] / data[:,2]
        ax.plot(xData, yData, '.', c=color, alpha=alpha)
    
    # PLOT FITTED FUNCTION
    xMin = min(xData)
    xMax = max(xData)
    xLength = xMax - xMin
    x       = np.linspace(xMin, xMax, num=1000)
    xLow    = np.linspace(xMin - extrapolLength*xLength, xMin, num=100)
    xHigh   = np.linspace(xMax, xMax + extrapolLength*xLength, num=100)
        
    fitValuesLow  = (1 - lapse - fit[3]) * options['sigmoidHandle'](xLow,  fit[0], fit[1]) + fit[3]
    fitValuesHigh = (1 - lapse - fit[3]) * options['sigmoidHandle'](xHigh, fit[0], fit[1]) + fit[3]
    fitValues     = (1 - lapse - fit[3]) * options['sigmoidHandle'](x,     fit[0], fit[1]) + fit[3]
    
    ax.plot(x,     fitValues,           c=color, lw=lineWidth, alpha=alpha)
    ax.plot(xLow,  fitValuesLow,  '--', c=color, lw=lineWidth, alpha=alpha)
    ax.plot(xHigh, fitValuesHigh, '--', c=color, lw=lineWidth, alpha=alpha)
    
    if xlim is None:
        xmin, xmax = xLow.min(), xHigh.max()
        xlim = (xmin, xmax)
    
    # Add dashed line at 50% and 100% - lapse-rate
    if plotAsymptote:
        ax.hlines(y=1./options['expN'], xmin=xlim[0], xmax=xlim[1], ls='-', lw=0.8, colors="gray")
        ax.hlines(y=1.0-lapse, xmin=xlim[0], xmax=xlim[1], ls='-', lw=0.8, colors="gray")
    ax.set(xlim=xlim)
        
    if plotCI:
        CIl = []; CIu = []
        ys = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x, fit[0], fit[1]) + fit[3]
        for i in range(len(ys)):
            [threshold, CI] = ps.getThreshold(result, ys[i])  # res, pc
            CIl.append(CI[2, 0]); CIu.append(CI[2, 1])  # 68% CI
        if not plotLapse:
            ys += fit[2] * options['sigmoidHandle'](x, fit[0], fit[1])
        ax.fill_betweenx(ys, CIl, CIu, color=color, alpha=0.2)

    if showImediate:
        plt.show()
    return axisHandle
