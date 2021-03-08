import numpy as np
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
import pickle as pickle
from .helpers import extract_hitran_data,line_ids_from_flux_calculator,line_ids_from_hitran,get_global_identifier, translate_molecule_identifier, get_molmass
import pdb as pdb
from astropy.table import Table
from astropy import units as un
import os
import urllib
import emcee
import pandas as pd
from astropy.convolution import Gaussian1DKernel, convolve
import json as json
import time
from IPython.display import display, Math
import corner
import matplotlib.pyplot as plt

def remove_burnin(presamples):
    postsamples=presamples[burnin:]
    return postsamples

def corner_plot(samples):
    parlabels=[ r"$\log(\ n_\mathrm{tot} [\mathrm{cm}^{-2}]\ )$",r"Temperature [K]", "$\log(\ {\Omega [\mathrm{rad}]}\ )$"]
    fig = corner.corner(samples,
                    labels=parlabels,
                    show_titles=True, title_kwargs={"fontsize": 12})

def trace_plot(samples,xr=[None,None]):
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    parlabels=[ r"$\log(\ n_\mathrm{tot} [\mathrm{cm}^{-2}]\ )$",r"Temperature [K]", "$\log(\ {\Omega [\mathrm{rad}]}\ )$"]
    ndims=3
    for i in range(ndims):
        ax = axes[i]
        ax.plot(samples[:,i], "k", alpha=0.3)    #0th walker, i'th dimension
        ax.set_ylabel(parlabels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_xlim(xr)
    axes[-1].set_xlabel("step number");


def find_best_fit(samples):
    parlabels=[ r"\log(\ n_\mathrm{tot} [\mathrm{cm}^{-2}]\ )",r"Temperature [K]", r"\log(\ {\Omega [\mathrm{rad}]}\ )"]
    paramkeys=['logN','T','logOmega']
    perrkeys=['logN_perr','T_perr','logOmega_perr']
    nerrkeys=['logN_nerr','T_nerr','logOmega_nerr']
    bestfit_dict={}
    for i in range(3):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], parlabels[i])
        display(Math(txt))
        bestfit_dict[paramkeys[i]]=mcmc[1]
        bestfit_dict[perrkeys[i]]=q[1]     
        bestfit_dict[nerrkeys[i]]=q[0]    

    return bestfit_dict
