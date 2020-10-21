import pandas as pd
import emcee
import corner
from astropy.constants import au,h,pc,c
from slabspec import *
from flux_calculator import *
from slab_fitter import * 
from scipy.optimize import minimize
import random
import numpy as np



def logposterior(theta, data, sigma, myrun,lognmin, lognmax, tmu, tsigma, logomegamin, logomegamax):

    """                                                                                       
    The natural logarithm of the joint posterior.                                             
                                                                                              
    Args:                                                                                     
        theta (list): a sample containing individual parameter values                        
        data (list): the set of data/observations                                             
        sigma (float): the standard deviation of the data points                              
        x (list): the abscissa values at which the data/model is defined                      
        lognmin (float) : minimum value for logn prior
        lognmax (float) : maximum value for logn prior
        tmu (float) : gaussian mean value for temp prior  
        tsigma (float) : gaussian sigma for temp prior
        logomegamin (float) : minimum value for logomega prior
        logomegamax (float) : minimum value for logomega prior                                                      
    """

    lp = logprior(theta,lognmin, lognmax, tmu, tsigma, logomegamin, logomegamax) # get the prior                                                      

    # if the prior is not finite return a probability of zero (log probability of -inf)       
    if not np.isfinite(lp):
        return -np.inf

    # return the likeihood times the prior (log likelihood plus the log prior)   
    return lp + loglikelihood(theta, data, sigma, myrun)
#-------------------------    
def logprior(theta, lognmin, lognmax, tmu, tsigma, logomegamin, logomegamax):
    """                                                                                       
    The natural logarithm of the prior probability.                                           
                                                                                              
    Args:                                                                                     
        theta (list): a sample containing individual parameter values
        lognmin (float) : minimum value for logn prior
        lognmax (float) : maximum value for logn prior
        tmu (float) : gaussian mean value for temp prior  
        tsigma (float) : gaussian sigma for temp prior
        logomegamin (float) : minimum value for logomega prior
        logomegamax (float) : minimum value for logomega prior                                                      
    """
    lp = 0.  #initialize log prior

    # unpack the model parameters from the list                                             
    logntot, temp, logomega = theta
#-------------------------    
#First parameter: logntot                                                          

    # set prior to 1 (log prior to 0) if in the range and zero (log prior to -inf)
    # outside the range       
    lp = 0. if lognmin < logntot < lognmax else -np.inf
#-------------------------    
#Second parameter: temperature
    # Gaussian prior on temp, except if T<=0                                                                   
    lp -= 0.5*((temp - tmu)/tsigma)**2   #Add log prior due to temperature to lp due to logn.

    # set prior to zero (log prior to -inf) if T<=0       
    if (temp <= 0):
        lp = -np.inf    
#-------------------------    
#Third parameter: Omega
    #Uniform prior on Omega, but with some cutoffs
    lpo = 0. if logomegamin < logomega < logomegamax else -np.inf

    lp += lpo #Add log prior due to omega to lp due to temperature,logn

    return lp
#-------------------------    
def loglikelihood(theta, data, sigma, myrun):

    """                                                                                       
    The natural logarithm of the joint likelihood.                                            
                                                                                              
    Args:                                                                                     
        theta (list): a sample containing individual parameter values                        
        data (list): the set of data/observations                                             
        sigma (float): the standard deviation of the data points                              
        x (list): the abscissa values at which the data/model is defined                                       
    """

    # unpack the model parameters from the tuple                                              
    logntot, temp, logomega = theta
    omega=10**logomega
    # evaluate the model (assumes that the model is defined above)           
    md = compute_fluxes(myrun, logntot, temp, omega)
    
    # return the log likelihood  
    return -0.5*np.sum(((md - data)/sigma)**2)


def rot_mcmc(data, minwave= 4.648, maxwave=5.018,lognmin=20, lognmax=24, tmu=750, tsigma=100, 
            logomegamin=-21,logomegamax=-12, plot=True):

    '''
    Inputs:
        data: a dictionary containing array entries where each element corresponds 
              to data from one emission line
              The dictionary should contain the keys:
                'molec_id'            : molecule id
                'local_iso_id'        : isotopologue id
                'Vp_HITRAN'           : upper V energy level
                'Vpp_HITRAN'          : lower V energy level
                'Qpp_HITRAN'          : P or R Branch emission line
                'lineflux'     : lineflux of emission line [W / m^2]
                'lineflux_err' : lineflux error of each emission line [W / m^2]
        lognmin             : minimum for logn prior
        lognmin (float)     : minimum value for logn prior
        lognmax (float)     : maximum value for logn prior
        tmu (float)         : gaussian mean value for temp prior  
        tsigma (float)      : gaussian sigma for temp prior
        logomegamin (float) : minimum value for logomega prior
        logomegamax (float) : minimum value for logomega prior                                                      
        plot=True           : bool to specify whether or not to plot 
    '''

    run = sf_run(data, minwave, maxwave)

    Nens = 250   # number of ensemble points    

    lognini = np.random.uniform(lognmin, lognmax, Nens) # initial logn points                                                                  
    tini = np.random.normal(tmu, tsigma, Nens) # initial t points                                                                                         
    logomegaini = np.random.uniform(logomegamin, logomegamax, Nens) # initial omegaini points                                    
    inisamples = np.array([lognini, tini, logomegaini]).T 

    ndims = inisamples.shape[1] # number of parameters/dimensions                                    
    Nburnin = 30   # number of burn-in samples                                                      
    Nsamples = 50  # number of final posterior samples 
    
    argslist = (run.lineflux, run.lineflux_err, run, lognmin, lognmax, tmu, tsigma, logomegamin, logomegamax)
                                                                       
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=argslist)
    sampler.run_mcmc(inisamples, Nsamples+Nburnin); 
    postsamples = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))


    if plot:
        fig = corner.corner(postsamples, labels=[ r"$\log(n_\mathrm{tot}$ $[m^{-2}])$",r"Temperature [K]", "$log(\Omega [\frac{m^2}{m^2}])$"],label_kwargs={"fontsize": 15})
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=12)

    return postsamples