import numpy as np
from astroquery.hitran import Hitran
from astropy import units as un
from astropy.constants import c, k_B, h, u
from molmass import Formula
import pdb as pdb
import pickle as pickle

def calc_solid_angle(radius,distance):
    '''
    Convenience function to calculate solid angle from radius and distance, assuming a disk shape.
    (For more complex shapes, user can calculate themselves).

    Parameters
    ----------
    radius : float
     radius value in AU
    distance : float
     distance value in parsec
    Returns
    ----------
    
    '''
    return np.pi*radius**2./(distance*206265.)**2.

def calc_radius(solid_angle,distance):
    '''
    Convenience function to calculate solid angle from radius and distance, assuming a disk shape.
    (For more complex shapes, user can calculate themselves).

    Parameters
    ----------
    solid_angle : float
     solid angle value in radians
    distance : float
     distance value in parsec
    Returns
    ----------
    
    '''
    return (distance*206265)*np.sqrt(solid_angle/np.pi)


def line_ids_from_flux_calculator(flux_calculator_output, line_id_dict):
    line_id_list=[]
    for i,myrow in enumerate(flux_calculator_output):
        line_id_key=(str(myrow['molec_id'])+str(myrow['local_iso_id']) + str(myrow['Vp_HITRAN'])+str(myrow['Vpp_HITRAN'])+ str(myrow['Qp_HITRAN'])+str(myrow['Qpp_HITRAN'])).replace(" ","")
        line_id_list.append(line_id_dict[line_id_key])
    
    line_id_arr=np.array(line_id_list)

    return line_id_arr

def line_ids_from_hitran(hitran_output):
    line_id_dict=pickle.load(open('line_id_dict.p','rb'))

    line_id_list=[]
    for i,myrow in enumerate(hitran_output):
        line_id_key=(str(myrow['molec_id'])+str(myrow['local_iso_id']) + str(myrow['Vp'])+str(myrow['Vpp'])+ str(myrow['Qp'])+str(myrow['Qpp'])).replace(" ","")
        line_id_list.append(line_id_dict[line_id_key])

    line_id_arr=np.array(line_id_list)

    return line_id_arr


def compute_thermal_velocity(molecule_name, temp):
    '''
    Compute the thermal velocity given a molecule name and temperature

    Parameters
    ---------
    molecule_name: string
      Molecule name (e.g., 'CO', 'H2O')
    temp : float
      Temperature at which to compute thermal velocity

    Returns
    -------
    v_thermal : float
       Thermal velocity (m/s)
    '''

    f=Formula(molecule_name)
    mu=f.isotope.mass*u.value   #kg

    return np.sqrt(k_B.value*temp/mu)   #m/s

def markgauss(x,mean=0, sigma=1., area=1):
    '''
    Compute a Gaussian function

    Parameters
    ----------
    x : float
      x values at which to calculate the Gaussian
    mean : float, optional
      mean of Gaussian
    sigma : float, optional
      standard deviation of Gaussian
    area : float, optional
      area of Gaussian curve

    Returns
    ---------
    Gaussian function evaluated at input x values

    '''

    norm=area
    u = ( (x-mean)/np.abs(sigma) )**2
    norm = norm / (np.sqrt(2. * np.pi)*sigma)
    f=norm*np.exp(-0.5*u)

    return f

def sigma_to_fwhm(sigma):
    '''
    Convert sigma to fwhm

    Parameters
    ----------
    sigma : float
       sigma of Gaussian distribution

    Returns
    ----------
    fwhm : float
       Full Width at Half Maximum of Gaussian distribution
    '''                        
    return  sigma*(2.*np.sqrt(2.*np.log(2.)))

def fwhm_to_sigma(fwhm):
    '''
    Convert fwhm to sigma

    Parameters
    ----------
    fwhm : float
       Full Width at Half Maximum of Gaussian distribution

    Returns
    ----------
    sigma : float
       sigma of Gaussian distribution
    '''                        

    return fwhm/(2.*np.sqrt(2.*np.log(2.)))

def wn_to_k(wn):
    '''                        
    Convert wavenumber to Kelvin

    Parameters
    ----------
    wn : AstroPy quantity
       Wavenumber including units

    Returns
    ---------
    energy : AstroPy quantity
       Energy of photon with given wavenumber

    '''              
    return wn.si*h*c/k_B

def extract_hitran_data(molecule_name, wavemin, wavemax, isotopologue_number=1, eupmax=None, aupmin=None,swmin=None,vup=None):
    '''                                                               
    Extract data from HITRAN 
    Primarily makes use of astroquery.hitran, with some added functionality specific to common IR spectral applications
    Parameters 
    ---------- 
    molecule_name : string
        String identifier for molecule, for example, 'CO', or 'H2O'
    wavemin: float
        Minimum wavelength of extracted lines (in microns)
    wavemax: float
        Maximum wavelength of extracted lines (in microns)                   
    isotopologue_number : float, optional
        Number representing isotopologue (1=most common, 2=next most common, etc.)
    eupmax : float, optional
        Maximum extracted upper level energy (in Kelvin)
    aupmin : float, optional
        Minimum extracted Einstein A coefficient
    swmin : float, optional
        Minimum extracted line strength
    Returns
    ------- 
    hitran_data : astropy table
        Extracted data
    '''

    #Convert molecule name to number
    M = get_molecule_identifier(molecule_name)

    #Convert inputs to astroquery formats
    min_wavenumber = 1.e4/wavemax
    max_wavenumber = 1.e4/wavemin

    #Extract hitran data using astroquery
    tbl = Hitran.query_lines(molecule_number=M,isotopologue_number=isotopologue_number,min_frequency=min_wavenumber / un.cm,max_frequency=max_wavenumber / un.cm)

    #Do some desired bookkeeping, and add some helpful columns
    tbl.rename_column('nu','wn')
    tbl['nu']=tbl['wn']*c.cgs.value   #Now actually frequency of transition
    tbl['eup_k']=(wn_to_k((tbl['wn']+tbl['elower'])/un.cm)).value
    tbl['wave']=1.e4/tbl['wn']       #Wavelength of transition, in microns
    tbl.rename_column('global_upper_quanta','Vp')
    tbl.rename_column('global_lower_quanta','Vpp')
    tbl.rename_column('local_upper_quanta','Qp')
    tbl.rename_column('local_lower_quanta','Qpp')

    #Extract desired portion of dataset
    ebool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    abool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    swbool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    vupbool = np.full(np.size(tbl), True, dtype=bool)  #default to True
    #Upper level energy
    if(eupmax is not None):
        ebool = tbl['eup_k'] < eupmax
    #Upper level A coeff
    if(aupmin is not None):
        abool = tbl['a'] > aupmin
    #Line strength
    if(swmin is not None):
        swbool = tbl['sw'] > swmin
    if(vup is not None):
        vupval = [np.int(val) for val in tbl['Vp']]
        vupbool=(np.array(vupval)==vup)

     #Combine
    extractbool = (abool & ebool & swbool & vupbool)
    hitran_data=tbl[extractbool]

    #Return astropy table
    return hitran_data

def get_global_identifier(molecule_name,isotopologue_number=1):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN *global* identifier number.
    For more info, see https://hitran.org/docs/iso-meta/ 
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.              
    isotopologue_number : int, optional
        The isotopologue number, from most to least common.                                                                              
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    G : int                                                                                                                            
        The HITRAN global identifier number.                                                                                        
    '''

    mol_isot_code=molecule_name+'_'+str(isotopologue_number)

    trans = { 'H2O_1':1, 'H2O_2':2, 'H2O_3':3, 'H2O_4':4, 'H2O_5':5, 'H2O_6':6, 'H2O_7':129,
               'CO2_1':7,'CO2_2':8,'CO2_3':9,'CO2_4':10,'CO2_5':11,'CO2_6':12,'CO2_7':13,'CO2_8':14,
               'CO2_9':121,'CO2_10':15,'CO2_11':120,'CO2_12':122,
               'O3_1':16,'O3_2':17,'O3_3':18,'O3_4':19,'O3_5':20,
               'N2O_1':21,'N2O_2':22,'N2O_3':23,'N2O_4':24,'N2O_5':25,
               'CO_1':26,'CO_2':27,'CO_3':28,'CO_4':29,'CO_5':30,'CO_6':31,
               'CH4_1':32,'CH4_2':33,'CH4_3':34,'CH4_4':35,
               'O2_1':36,'O2_2':37,'O2_3':38,
               'NO_1':39,'NO_2':40,'NO_3':41,
               'SO2_1':42,'SO2_2':43,
               'NO2_1':44,
               'NH3_1':45,'NH3_2':46,
               'HNO3_1':47,'HNO3_2':117,
               'OH_1':48,'OH_2':49,'OH_3':50,
               'HF_1':51,'HF_2':110,
               'HCl_1':52,'HCl_2':53,'HCl_3':107,'HCl_4':108,
               'HBr_1':54,'HBr_2':55,'HBr_3':111,'HBr_4':112,
               'HI_1':56,'HI_2':113,
               'ClO_1':57,'ClO_2':58,
               'OCS_1':59,'OCS_2':60,'OCS_3':61,'OCS_4':62,'OCS_5':63,
               'H2CO_1':64,'H2CO_2':65,'H2CO_3':66,
               'HOCl_1':67,'HOCl_2':68,
               'N2_1':69,'N2_2':118,
               'HCN_1':70,'HCN_2':71,'HCN_3':72,
               'CH3Cl_1':73,'CH3CL_2':74,
               'H2O2_1':75,
               'C2H2_1':76,'C2H2_2':77,'C2H2_3':105,
               'C2H6_1':78,'C2H6_2':106,
               'PH3_1':79,
               'COF2_1':80,'COF2_2':119,
               'SF6_1':126,
               'H2S_1':81,'H2S_2':82,'H2S_3':83,
               'HCOOH_1':84,
               'HO2_1':85,
               'O_1':86,
               'ClONO2_1':127,'ClONO2_2':128,
               'NO+_1':87,
               'HOBr_1':88,'HOBr_2':89,
               'C2H4_1':90,'C2H4_2':91,
               'CH3OH_1':92,
               'CH3Br_1':93,'CH3Br_2':94,
               'CH3CN_1':95,
               'CF4_1':96,
               'C4H2_1':116,
               'HC3N_1':109,
               'H2_1':103,'H2_2':115,
               'CS_1':97,'CS_2':98,'CS_3':99,'CS_4':100,
               'SO3_1':114,
               'C2N2_1':123,
               'COCl2_1':124,'COCl2_2':125}
 
    return trans[mol_isot_code]

def get_molmass(molecule_name,isotopologue_number=1):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding molecular mass, in amu
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.              
    isotopologue_number : int, optional
        The isotopologue number, from most to least common.                                                                              
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    mu : float                                                                                                                            
        Molecular mass in amu
    '''

    mol_isot_code=molecule_name+'_'+str(isotopologue_number)

#https://hitran.org/docs/iso-meta/

    mass = { 'H2O_1':18.010565, 'H2O_2':20.014811, 'H2O_3':19.01478, 'H2O_4':19.01674, 
               'H2O_5':21.020985, 'H2O_6':20.020956, 'H2O_7':20.022915,
               'CO2_1':43.98983,'CO2_2':44.993185,'CO2_3':45.994076,'CO2_4':44.994045,
               'CO2_5':46.997431,'CO2_6':45.9974,'CO2_7':47.998322,'CO2_8':46.998291,
               'CO2_9':45.998262,'CO2_10':49.001675,'CO2_11':48.001646,'CO2_12':47.0016182378,
               'O3_1':47.984745,'O3_2':49.988991,'O3_3':49.988991,'O3_4':48.98896,'O3_5':48.98896,
               'N2O_1':44.001062,'N2O_2':44.998096,'N2O_3':44.998096,'N2O_4':46.005308,'N2O_5':45.005278,
               'CO_1':27.994915,'CO_2':28.99827,'CO_3':29.999161,'CO_4':28.99913,'CO_5':31.002516,'CO_6':30.002485,
               'CH4_1':16.0313,'CH4_2':17.034655,'CH4_3':17.037475,'CH4_4':18.04083,
               'O2_1':31.98983,'O2_2':33.994076,'O2_3':32.994045,
               'NO_1':29.997989,'NO_2':30.995023,'NO_3':32.002234,
               'SO2_1':63.961901,'SO2_2':65.957695,
               'NO2_1':45.992904,'NO2_2':46.989938,
               'NH3_1':17.026549,'NH3_2':18.023583,
               'HNO3_1':62.995644,'HNO3_2':63.99268,
               'OH_1':17.00274,'OH_2':19.006986,'OH_3':18.008915,
               'HF_1':20.006229,'HF_2':21.012404,
               'HCl_1':35.976678,'HCl_2':37.973729,'HCl_3':36.982853,'HCl_4':38.979904,
               'HBr_1':79.92616,'HBr_2':81.924115,'HBr_3':80.932336,'HBr_4':82.930289,
               'HI_1':127.912297,'HI_2':128.918472,
               'ClO_1':50.963768,'ClO_2':52.960819,
               'OCS_1':59.966986,'OCS_2':61.96278,'OCS_3':60.970341,'OCS_4':60.966371,'OCS_5':61.971231, 'OCS_6':62.966136,
               'H2CO_1':30.010565,'H2CO_2':31.01392,'H2CO_3':32.014811,
               'HOCl_1':51.971593,'HOCl_2':53.968644,
               'N2_1':28.006148,'N2_2':29.003182,
               'HCN_1':27.010899,'HCN_2':28.014254,'HCN_3':28.007933,
               'CH3Cl_1':49.992328,'CH3CL_2':51.989379,
               'H2O2_1':34.00548,
               'C2H2_1':26.01565,'C2H2_2':27.019005,'C2H2_3':27.021825,
               'C2H6_1':30.04695,'C2H6_2':31.050305,
               'PH3_1':33.997238,
               'COF2_1':65.991722,'COF2_2':66.995083,
               'SF6_1':145.962492,
               'H2S_1':33.987721,'H2S_2':35.983515,'H2S_3':34.987105,
               'HCOOH_1':46.00548,
               'HO2_1':32.997655,
               'O_1':15.994915,
               'ClONO2_1':96.956672,'ClONO2_2':98.953723,
               'NO+_1':29.997989,
               'HOBr_1':95.921076,'HOBr_2':97.919027,
               'C2H4_1':28.0313,'C2H4_2':29.034655,
               'CH3OH_1':32.026215,
               'CH3Br_1':93.941811,'CH3Br_2':95.939764,
               'CH3CN_1':41.026549,
               'CF4_1':87.993616,
               'C4H2_1':50.01565,
               'HC3N_1':51.010899,
               'H2_1':2.01565,'H2_2':3.021825,
               'CS_1':43.971036,'CS_2':45.966787,'CS_3':44.974368,'CS_4':44.970399,
               'SO3_1':79.95682,
               'C2N2_1':52.006148,
               'COCl2_1':97.9326199796,'COCl2_2':99.9296698896,
               'CS2_1':75.94414,'CS2_2':77.93994,'CS2_3':76.943256,'CS2_4':76.947495}
 
    return mass[mol_isot_code]


#Code from Nathan Hagen
#https://github.com/nzhagen/hitran
def translate_molecule_identifier(M):
    '''                                                                                                            
    For a given input molecule identifier number, return the corresponding molecular formula.                      
                                                                                                                   
    Parameters                                                                                                     
    ----------                                                                                                     
    M : int                                                                                                        
        The HITRAN molecule identifier number.                                                                     
                                                                                                                   
    Returns                                                                                                        
    -------                                                                                                        
    molecular_formula : str                                                                                        
        The string describing the molecule.                                                                        
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    return(trans[str(M)])

#Code from Nathan Hagen
#https://github.com/nzhagen/hitran
def get_molecule_identifier(molecule_name):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN molecule identifier number.                                   
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.                                                                                            
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    M : int                                                                                                                            
        The HITRAN molecular identifier number.                                                                                        
    '''

    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    ## Invert the dictionary.                                                                                                          
    trans = {v:k for k,v in trans.items()}
    return(int(trans[molecule_name]))
