import numpy as np
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
import pickle as pickle
from .helpers import extract_hitran_data,line_ids_from_flux_calculator,line_ids_from_hitran,get_global_identifier, translate_molecule_identifier
import pdb as pdb
from astropy.table import Table
from astropy import units as un
import os
import urllib
import pandas as pd
from astropy.convolution import Gaussian1DKernel, convolve

class sf_run():
    def __init__(self,data):
        
        #will want to modify following for isotopologues, other molecules.  probably its own function.
        self.qdata=pd.read_csv('https://hitran.org/data/Q/q26.txt',sep=' ',skipinitialspace=True,names=['temp','q'],header=None)

        self.wn0=data['wn']*1e2 # now m-1
        self.aup=data['a']
        self.eup=(data['elower']+data['wn'])*1e2 #now m-1
        self.gup=data['gup']
        self.eup_k=data['eup_k']
        self.elower=data['elower']
        self.molec_id=data['molec_id']
        self.local_iso_id=data['local_iso_id']
        self.qpp = data['Qpp_HITRAN']
        self.qp = data['Qp_HITRAN']
        self.vp = data['Vp_HITRAN'] 
        self.vpp = data['Vpp_HITRAN']
        self.lineflux=data['lineflux']
        self.lineflux_err=data['lineflux_err']
        self.nlines = len(self.lineflux)
        self.global_id=return_global_ids(self)  #Determine HITRAN global ids (molecule + isotope) for each line
        self.unique_globals = np.unique(self.global_id)
        self.qdata_dict=get_qdata(self.unique_globals)

#------------------------------------------------------------------------------------                                     
def get_hitran_from_flux_calculator(data,hitran_data):

    #Define line_id_dictionary using hitran_data
    line_id_dict={}
    for i,myrow in enumerate(hitran_data):
        line_id_key=(str(myrow['molec_id'])+str(myrow['local_iso_id']) + str(myrow['Vp'])+str(myrow['Vpp'])+ str(myrow['Qp'])+str(myrow['Qpp'])).replace(" ","")
        line_id_dict[line_id_key]=i

    #Get a set of line_ids using hitran_data and actual data
    line_ids=line_ids_from_flux_calculator(data, line_id_dict)

    #Get appropriate partition function data (will need to fix to account for isotopologues, possibly different molecules)
    qdata = pd.read_csv('https://hitran.org/data/Q/q26.txt',sep=' ',skipinitialspace=True,names=['temp','q'],header=None)

    hitran_dict={'qdata':qdata,'line_ids':line_ids,'hitran_data':hitran_data}

    return hitran_dict

#------------------------------------------------------------------------------------                                     
def compute_fluxes(myrun,logn,temp,omega):
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

#Use line_ids to extract relevant HITRAN data
    wn0=myrun.wn0
    aup=myrun.aup
    eup=myrun.eup
    gup=myrun.gup
    eup_k=myrun.eup_k
    elower=myrun.elower 

#Compute partition function - will need to address isotopes/different molecules later
    q=compute_partition_function_co(temp,myrun.qdata)
#Begin calculations                                                                                                       
    afactor=((aup*gup*n_col)/(q*8.*np.pi*(wn0)**3.)) #mks                                                                 
    efactor=h.value*c.value*eup/(k_B.value*temp)
    wnfactor=h.value*c.value*wn0/(k_B.value*temp)
    phia=1./(deltav*np.sqrt(2.0*np.pi))
    efactor2=eup_k/temp
    efactor1=elower*1.e2*h.value*c.value/k_B.value/temp
    tau0=afactor*(np.exp(-1.*efactor1)-np.exp(-1.*efactor2))*phia  #Avoids numerical issues at low T

    w0=1.e6/wn0

    dvel=0.1e0    #km/s
    nvel=1001
    vel=(dvel*(np.arange(0,nvel)-500.0))*1.e3     #now in m/s   

#Now loop over transitions and velocities to calculate flux
    tau=np.exp(-vel**2./(2.*deltav**2.))*np.vstack(tau0)

#Create array to hold line fluxes (one flux value per line)
    nlines=np.size(tau0)
    f_arr=np.zeros([nlines,nvel])     #nlines x nvel       
    lineflux=np.zeros(nlines)

    for i in range(nlines):  #I might still be able to get rid of this loop
        f_arr[i,:]=2*h.value*c.value*wn0[i]**3./(np.exp(wnfactor[i])-1.0e0)*(1-np.exp(-tau[i,:]))*si2jy*omega
        lineflux_jykms=np.sum(f_arr[i,:])*dvel
        lineflux[i]=lineflux_jykms*1e-26*1.*1e5*(1./(w0[i]*1e-4))    #mks

    return lineflux

def get_qdata(id_array):
    q_dict={}
    for myid in id_array:
        qurl='https://hitran.org/data/Q/'+'q'+str(myid)+'.txt'
        handle = urllib.request.urlopen(qurl)
        qdata = pd.read_csv(handle,sep=' ',skipinitialspace=True,names=['temp','q'],header=None)
        q_dict.update({str(myid):qdata['q']})
    return q_dict

def compute_partition_function_co(temp,qdata,isotopologue_number=1):
    q=np.interp(temp,qdata['temp'],qdata['q'])  
    return q

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

#---------------------
def return_global_ids(self):
    global_id = np.array([get_global_identifier(translate_molecule_identifier(self.molec_id[i]), isotopologue_number=self.local_iso_id[i]) for i in np.arange(self.nlines)])
    return global_id
