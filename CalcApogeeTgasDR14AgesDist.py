"""
CALCULATE APOGEE-TGAS DISTANCES AND AGES
Uses Bayesian method applied to spectro-photo-astrom-seismo data, accounting for dust
"""
import numpy as np
import pandas as pd
import CalcSpectroPhotoAstromSeismoDistGrid as cspas
import CoordTrans as ct
import matplotlib.pyplot as plt
import time
from pathos.multiprocessing import Pool

#%% READ DATA

#header=np.hstack([["location_id","ra","dec","vlos","evlos","j","ej","h","eh","k","ek","teff","teff_err","logg",
#                   "logg_err","m_h","m_h_err","alpha_m","alpha_m_err","ra_gaia","dec_gaia","ra_error",
#                   "dec_error","parallax","parallax_error","pmra","pmra_error","pmdec","pmdec_error","annMass",
#                   "eAnnMass","lncanmass","elncanmass","lncanage","elncanage","kepmass","kepmass68L",
#                   "kepmass68U","kepmass95L","kepmass95U","kepage","kepage68L","kepage68U","kepage95L",
#                   "kepage95U","kepevstate"]])
datafile = "../data/apogee/APOGEE_TGAS_DR14_supp_keplercannonann_masses_ages.csv"
data     = pd.read_csv(datafile,header=0,index_col=0) 
nstars   = len(data)  
                
print("The number of stars in the APOGEE-TGAS DR14 sample is:")
print(nstars) 
print(" ") 
        
# TGAS Ra, TGAS Dec, TGAS parallax, APOGEE J, APOGEE H, APOGEE K, APOGEE Te, 
# APOGEE logg, APOGEE [M/H], ANN mass
Obs  = np.column_stack((data["ra_gaia"],data["dec_gaia"],data["parallax"],
                        data["j"],data["h"],data["k"],data["teff"],
                        data["logg"],data["m_h"],data["annMass"]))

# TGAS Ra error, TGAS Dec error, TGAS parallax error, APOGEE J error, 
# APOGEE H error, APOGEE K error, APOGEE Te error, APOGEE logg error, 
# APOGEE [M/H] error, ANN mass error
eObs = np.column_stack((data["ra_error"],data["dec_error"],
                        data["parallax_error"],data["ej"],data["eh"],
                        data["ek"],data["teff_err"],data["logg_err"],
                        data["m_h_err"],data["eAnnMass"]))  
                    
#%% CALCULATE DISTANCES FOR ALL STARS 
# Instantiate class for estimating distances
solpos = np.array([8.3,0.014,-14.0,12.24+238.,7.25])
dust   = True
taum   = 13.1
spas   = cspas.SpectroPhotoAstromSeismoDist(solpos,dust,taum)

# Whether to be silent 
silent  = True

# Which data to use in fit(appmag,col,logg,teff,feh,mass,parallax)
usedata = np.array([True,True,False,False,False,True,True])

# Whether to use galaxy prior    
usegalprior  = True
                
#%% Determine posterior probability function and samples of mass, age, metallicity, distance for all stars
nsamp = 1000

def calcAgeDist(jstars):

#for jstars in range(nstars):
    
    #timestart = time.time()
     
    print("STAR "+str(jstars)+":")
    
    # Calculate Galactic longitude and latitude
    xe   = np.column_stack((Obs[jstars,0],Obs[jstars,1],1.))
    xg   = ct.EquatorialToGalactic(xe) 
    obsl = xg[0,0] # rad
    obsb = xg[0,1] # rad
    
    # Coordinates of star
    # obsStar  - varpi,Jmag,Hmag,Kmag,teff,logg,mh,mass
    # eobsStar - evarpi,eJmag,eHmag,eKmag,eteff,elogg,emh,emass  
      
    obsStar   = np.hstack((Obs[jstars,2:10]))
    eobsStar  = np.hstack((eObs[jstars,2:10]))    
    obsvarpi  = obsStar[0]
    eobsvarpi = eobsStar[0]
              
    # Calculate grid of posterior probabilities
    nsig         = 3.
    nmass        = 30
    ns           = 100
    varpimin     = obsvarpi-nsig*eobsvarpi
    if (varpimin<0.):
        varpimin = 0.001
    varpimax     = obsvarpi+nsig*eobsvarpi
    varpigrid    = np.linspace(varpimin,varpimax,ns)
    sgrid        = 1./varpigrid
    sgrid        = np.sort(sgrid)
    
    agepost,mhpost,masspost,spost,log10agepost,dmpost,ppost = \
        spas.pPostGrid(obsStar,eobsStar,obsl,obsb,nsig,nmass,sgrid,usegalprior,usedata,silent)

    # Calculate marginalized posterior probabilities and samples
    log10agesamp,dmsamp = \
        spas.calcMargAgeDist(agepost,mhpost,masspost,spost,log10agepost,
                            dmpost,ppost,nsamp)
    log10agemoments = np.array([np.mean(log10agesamp),np.std(log10agesamp)])
    dmmoments       = np.array([np.mean(dmsamp),np.std(dmsamp)])

    #timeend   = time.time()
    #timetaken = timeend-timestart
    #print("Run took "+ str(timetaken) + "s.")
    #print(" ") 

    return(log10agemoments,dmmoments)
    
#res = map(calcAgeDist, range(nstars))

#%% Parallelized

timestart = time.time()
num_cores = 3
pool      = Pool(processes=num_cores)
res       = pool.map(calcAgeDist, range(nstars))
pool.close() 
timeend   = time.time()
timetaken = timeend-timestart
print("Run took "+ str(timetaken) + "s.")
print(" ") 

#%%
fitdata = ["_appmag","_col","_logg","_teff","feh","mass","parallax"]
log10agemomentsfile  = "../results/apogee/dist/log10agemoments_apogeetgas"
dmmomentsfile        = "../results/apogee/dist/dmmoments_apogeetgas"
if (usegalprior==True):
    fileinfo = "_usegalprior"
else:
    fileinfo = "_nogalprior"
if (usedata[0]==True):
    fileinfo += "_appmag"
if (usedata[1]==True):
    fileinfo += "_col"
if (usedata[2]==True):
    fileinfo += "_logg"
if (usedata[3]==True):
    fileinfo += "_teff"
if (usedata[4]==True):
    fileinfo += "_feh"
if (usedata[5]==True):
    fileinfo += "_mass"
if (usedata[6]==True):
    fileinfo += "_parallax"
log10agemomentsfile  += fileinfo+".txt"
dmmomentsfile        += fileinfo+".txt"
print(log10agemomentsfile)
print(dmmomentsfile)

log10agemoments  = np.zeros([nstars,2])
dmmoments        = np.zeros([nstars,2])
for jstars in range(nstars):
    log10agemoments[jstars,:] = res[jstars][:][0]
    dmmoments[jstars,:]       = res[jstars][:][1]
np.savetxt(log10agemomentsfile,log10agemoments)
np.savetxt(dmmomentsfile,dmmoments)

#%% CREATE PLOTS

dmmin       = 5
dmmax       = 22

# Check distances
plt.rc('font',family='serif')  
fig0 = plt.figure(figsize=(5,5))        
plt.scatter(5.*np.log10(1000./Obs[:,2])-5.,dmmoments[:,0],s=30, facecolors='none', edgecolors='r')
plt.plot([dmmin,dmmax],[dmmin,dmmax],':k')
plt.title(fileinfo,fontsize=16)
plt.xlim(dmmin,dmmax)
plt.ylim(dmmin,dmmax)
plt.xlabel(r"$\mu_{\mathrm{TGAS}}$ (mag)", fontsize=14)
plt.ylabel(r"$\mu_{\mathrm{BNN}}$ (mag)", fontsize=14)
fig0.savefig("../plots/apogee/dist/apogee_bnn_dmcomp"+fileinfo+".eps",fmt="eps")
