"""
Main script for reading and saving reformated experimental data 

Last update: April 2024
@author: lynnschmittwilken
"""

import numpy as np
import pandas as pd
from functions import reformat_data, get_lapse_rate, load_all_data

vps = ["ls", "mm", "jv", "ga", "sg", "fd"]
datadir = "./experimental_data/"

####################################
#            Read data             #
####################################
# Load data of all vps
data = load_all_data(datadir, vps)


####################################
#       Reformat pooled data       #
####################################
noise_conds = np.unique(data["noise"])
edge_conds = np.unique(data["edge_width"])

df_list = []
for n in noise_conds:
    for e in edge_conds:
        # Reformat pooled data
        contrasts, ncorrect, ntrials = reformat_data(data, n, e)
        lamb = get_lapse_rate(contrasts, ncorrect, ntrials)

        df = pd.DataFrame({
            "noise": [n,]*len(contrasts),
            "edge": [e,]*len(contrasts),
            "contrasts": contrasts,
            "ncorrect": ncorrect,
            "ntrials": ntrials,
            "lambda": [lamb,]*len(contrasts),
            })
        df_list.append(df)
df = pd.concat(df_list).reset_index(drop=True)


####################################
#          Save dataframe          #
####################################
headers = df.columns.tolist()
np.savetxt(datadir + "expdata_pooled.txt", df, fmt='%s', header=' '.join(headers), comments='')
