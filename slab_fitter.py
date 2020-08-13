import numpy as np
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
import pickle as pickle
from .helpers import extract_hitran_data,line_ids_from_flux_calculator,line_ids_from_hitran,get_global_identifier
import pdb as pdb
from astropy.table import Table
from astropy import units as un
import os
import urllib
import pandas as pd
from astropy.convolution import Gaussian1DKernel, convolve

#------------------------------------------------------------------------------------                                     
def compute_fluxes(line_ids,logn,temp,omega):
    '''
    Compute line fluxes for a slab model with given column density, temperature, and solid angle=area/d^2 
    '''
    n_col=10**logn

    isot=1   #Will need to fix later
    si2jy=1e26   #SI to Jy flux conversion factor

#If local velocity field is not given, assume sigma given by thermal velocity
#Will eventually need to account for isotopologues

    mu=u.value*28.01  #12CO
    deltav=np.sqrt(k_B.value*temp/mu)   #m/s 

#Read HITRAN data
    hitran_data=extract_hitran_data('CO',4.5,5.2,isotopologue_number=isot)

#Use line_ids to extract relevant HITRAN data
    wn0=hitran_data['wn'][line_ids]*1e2 # now m-1
    aup=hitran_data['a'][line_ids]
    eup=(hitran_data['elower'][line_ids]+hitran_data['wn'][line_ids])*1e2 #now m-1
    gup=hitran_data['gp'][line_ids]

#Compute partition function - will want to write local version of this.
    q=compute_partition_function('CO',temp,isot)

#Begin calculations                                                                                                       
    afactor=((aup*gup*n_col)/(q*8.*np.pi*(wn0)**3.)) #mks                                                                 
    efactor=h.value*c.value*eup/(k_B.value*temp)
    wnfactor=h.value*c.value*wn0/(k_B.value*temp)
    phia=1./(deltav*np.sqrt(2.0*np.pi))
    efactor2=hitran_data['eup_k'][line_ids]/temp
    efactor1=hitran_data['elower'][line_ids]*1.e2*h.value*c.value/k_B.value/temp
    tau0=afactor*(np.exp(-1.*efactor1)-np.exp(-1.*efactor2))*phia  #Avoids numerical issues at low T

    w0=1.e6/wn0

    dvel=0.1e0    #km/s
    nvel=1001
    vel=(dvel*(np.arange(0,nvel)-500.0))*1.e3     #now in m/s   

    fthin=aup*gup*n_col*h.value*c.value*wn0/(q*4.*np.pi)*np.exp(-efactor)*omega # Energy/area/time, mks                   
#Now loop over transitions and velocities to calculate flux                                                               
    nlines=np.size(tau0)
    tau=np.zeros([nlines,nvel])
    wave=np.zeros([nlines,nvel])
    for ha,mytau in enumerate(tau0):
        for ka, myvel in enumerate(vel):
            tau[ha,ka]=tau0[ha]*np.exp(-vel[ka]**2./(2.*deltav**2.))

#Now interpolate over wavelength space so that all lines can be added together                                            
    w_arr=wave            #nlines x nvel                                                                                  
    f_arr=w_arr-w_arr     #nlines x nvel                                                                                  
#Create array to hold line fluxes (one flux value per line)
    lineflux=np.zeros(nlines)
    for i in range(nlines):
        f_arr[i,:]=2*h.value*c.value*wn0[i]**3./(np.exp(wnfactor[i])-1.0e0)*(1-np.exp(-tau[i,:]))*si2jy*omega
        lineflux_jykms=np.sum(f_arr[i,:])*dvel
        lineflux[i]=lineflux_jykms*1e-26*1.*1e5*(1./(w0[i]*1e-4))    #mks

    return lineflux


def compute_partition_function(molecule_name,temp,isotopologue_number=1):
    '''                                                                                                                                       
    For a given input molecule name, isotope number, and temperature, return the partition function Q
                                                                                                                                              
    Parameters                                                                                                                                
    ----------                                                                                                                                
    molecule_name : string
        The molecule name string (e.g., 'CO', 'H2O')
    temp : float
        The temperature at which to compute the partition function
    isotopologue_number : float, optional
        Isotopologue number, with 1 being most common, etc. Defaults to 1.

    Returns                                                                                                                                   
    -------                                                                                                                                   
    q : float
      The partition function
    '''

    G=get_global_identifier(molecule_name, isotopologue_number=isotopologue_number)
    qurl='https://hitran.org/data/Q/'+'q'+str(G)+'.txt'
    handle = urllib.request.urlopen(qurl)
    qdata = pd.read_csv(handle,sep=' ',skipinitialspace=True,names=['temp','q'],header=None)

#May want to add code with local file access
#    pathmod=os.path.dirname(__file__)
#    if not os.path.exists(qfilename):  #download data from internet
       #get https://hitran.org/data/Q/qstr(G).txt

    q=np.interp(temp,qdata['temp'],qdata['q'])
    return q


#Make this its own function
def make_rotation_diagram(lineparams,modelfluxes=None):
    '''
    Take ouput of make_spec and use it to compute rotation diagram parameters.

    Parameters
    ---------
    lineparams: dictionary
        dictionary output from make_spec

    Returns
    --------
    rot_table: astropy Table
        Table of x and y values for rotation diagram.  

    '''
    x=lineparams['eup_k']
    y=np.log(lineparams['lineflux']/(lineparams['wn']*lineparams['gup']*lineparams['a']))
    rot_table = Table([x, y], names=('x', 'y'),  dtype=('f8', 'f8'))
    rot_table['x'].unit = 'K'        

    if(modelfluxes is not None):
        rot_table['modely']=np.log(modelfluxes/(lineparams['wn']*lineparams['gup']*lineparams['a']))

    return rot_table
