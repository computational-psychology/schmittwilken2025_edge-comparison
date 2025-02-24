This is the code used to produce the results and visualizations published in

Schmittwilken, L., & Maertens, M. (under submission). Ocular drift shakes the stationary view on pattern vision

## Setup

Install all the libraries in  `requirements.txt`.
```bash
pip install -r requirements.txt
```
Note: we have used an older version of `python-psignifit` here, which is not available anymore. Therefore, we decided to add it to the repo directly in the folder [psignifit](psignifit). You can find information on the newest version of psignifit [here](https://github.com/wichmann-lab/python-psignifit).

## Description
The repository contains the following:

- The empirical edge sensitivity data by [Schmittwilken, Wichmann, & Maertens (2024)](https://doi.org/10.1016/j.visres.2024.108450) who probed sensitivity to Cornsweet edges in different 2d noise patterns: [experimental_data](experimental_data)

- A Jupyter notebook which contains all simulations related to testing the effect of ocular drift on the stimulus spectra: [heuristic_test](heuristic_test)

- Additional Jupyter notebooks with toy models as well as additional information about the spatial and temporal filters used: [jupyter_notebooks](jupyter_notebooks)

- All scripts related to implementing and optimizing the mechanistic models as well as visualizing their performances. In addition, it contains pickle-files with the optimized spatial and active models that we report in the manuscript: [mechanistic_models](mechanistic_models)

- An old version of `python-psignifit`: [psignifit](psignifit)

## Authors and acknowledgment
Code written by Lynn Schmittwilken (l.schmittwilken@tu-berlin.de)
