import numpy as np
from astropy.io import fits
from astropy.constants import c,h, k_B, G, M_sun, au, pc, u
import pickle as pickle
from helpers import extract_hitran_data,line_ids_from_flux_calculator,line_ids_from_hitran,get_global_identifier,translate_molecule_identifier
import pdb as pdb
from astropy.table import Table
from astropy import units as un
import os
import urllib
import pandas as pd
from astropy.convolution import Gaussian1DKernel, convolve

class sf_run():
    def __init__(self,data, minwave, maxwave):

        # this info is required of the user
        # for each emission line: lineflux, lineflux_err,
        # Molecule id, local isotopologue id, 
        # upper and lower local quanta (Qp, Qpp),
        # upper and lower global quanta (Vp, Vpp)
        # each index = i of each array corresponds to the same emission line
        self.lineflux=data['lineflux']
        self.lineflux_err=data['lineflux_err']
        self.molec_id=data['molec_id']
        self.local_iso_id=data['local_iso_id']
        self.qpp = data['Qpp_HITRAN']
        self.qp = data['Qp_HITRAN']
        self.vp = data['Vp_HITRAN']
        self.vpp = data['Vpp_HITRAN']
        
        num_line = len(self.lineflux)

        # identify unique isotopologues so we only have to query 
        # hitran once per global id to get partition function
        unique_isotopologues = list(set(zip(self.molec_id, self.local_iso_id)))
        

        # identify global id for each measurement for later reference to partition function
        self.global_id = np.array([get_global_identifier(translate_molecule_identifier(self.molec_id[i]), \
                                    isotopologue_number=self.local_iso_id[i])\
                                    for i in np.arange(num_line)])


        # collect partition function and hitran data once for each unique global id
        
        self.hitran_qdata = {}

        for mol,iso in unique_isotopologues:
            ident = get_global_identifier(translate_molecule_identifier(mol), isotopologue_number=iso)
            self.hitran_qdata[ident] = {}
            self.hitran_qdata[ident]['qdata'] = compute_partition_function(translate_molecule_identifier(mol), isotopologue_number=iso)
            self.hitran_qdata[ident]['hitran'] = extract_hitran_data(translate_molecule_identifier(mol), minwave, maxwave, isotopologue_number=iso)

        # add correct corresponding hitran info to arrays
        
        # initialize empty arrays. each index corresponds to the 
        # emission line at the same index
        self.wn0 = np.zeros(num_line, dtype=float)*np.nan
        self.aup = np.zeros(num_line, dtype=float)*np.nan
        self.eup = np.zeros(num_line, dtype=float)*np.nan
        self.gup = np.zeros(num_line, dtype=float)*np.nan
        self.eup_k = np.zeros(num_line, dtype=float)*np.nan
        self.elower = np.zeros(num_line, dtype=float)*np.nan
        
        # assign values for each emission line
        for i in np.arange(num_line):
            
            # hd = hitran data for isotopologue
            hd = self.hitran_qdata[self.global_id[i]]['hitran']

            # correctLine checks each field
            correctLine = (self.molec_id[i] == hd['molec_id']) & \
                          (self.local_iso_id[i] == hd['local_iso_id']) & \
                          (self.qpp[i] ==hd['Qpp']) & \
                          (self.qp[i] == hd['Qp']) & \
                          (self.vp[i] == hd['Vp']) & \
                          (self.vpp[i] == hd['Vpp'])

            index = [j for j, x in enumerate(correctLine) if x]


            h = hd[index]
            
            self.wn0[i]=h['wn']*1e2 # now m-1
            self.aup[i]=h['a']
            self.eup[i]=(h['elower']+self.wn0[i])*1e2 #now m-1
            self.gup[i]=h['gp']
            self.eup_k[i]=h['eup_k']
            self.elower[i]=h['elower']

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

# Get partition function at temp for each line
    q=get_qdata(myrun, temp)
    
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

# Defining get_qdata to interpolate for each emission line the partition function at temp
def get_qdata(myrun, temp):
    qdata = np.zeros(len(myrun.lineflux)) * np.nan

    for i in np.arange(len(myrun.lineflux)):
        qdata[i] = np.interp(temp, \
                myrun.hitran_qdata[myrun.global_id[i]]['qdata']['temp'], \
                myrun.hitran_qdata[myrun.global_id[i]]['qdata']['q'])

    return qdata

def compute_partition_function_co(temp,qdata,isotopologue_number=1):
    q=np.interp(temp,qdata['temp'],qdata['q'])  
    return q

def compute_partition_function(molecule_name,isotopologue_number=1):
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

    return qdata


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
