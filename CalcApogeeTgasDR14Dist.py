"""
CALCULATE APOGEE-TGAS DISTANCES
Uses Bayesian method applied to astrometric data to estimate distances
"""
import numpy as np
import pandas as pd
import CalcAstromDistGrid as cad
import CoordTrans as ct
import matplotlib.pyplot as plt
import time

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
        
# TGAS Ra, TGAS Dec, TGAS parallax, APOGEE J, APOGEE H, APOGEE K 
Obs  = np.column_stack((data["ra_gaia"],data["dec_gaia"],data["parallax"],
                        data["j"],data["h"],data["k"]))

# TGAS Ra error, TGAS Dec error, TGAS parallax error, APOGEE J error, 
# APOGEE H error, APOGEE K error
eObs = np.column_stack((data["ra_error"],data["dec_error"],
                        data["parallax_error"],data["ej"],data["eh"],
                        data["ek"]))  
                    
#%% CALCULATE DISTANCES FOR ALL STARS 
                    
# Instantiate class for estimating distances
solpos = np.array([8.3,0.014,-14.0,12.24+238.,7.25])
dust   = True
ad     = cad.AstromDist(solpos,dust)

# Whether to use galaxy prior    
usegalprior = False
                
# Determine posterior probability function and samples of distance and absolute
# J-band magnitude for all stars
nsamp     = 1000
dmmoments = np.zeros([nstars,2])

for jstars in range(nstars):
     
    timestart = time.time()
    print("STAR "+str(jstars)+":")
    
    # Calculate Galactic longitude and latitude
    xe   = np.column_stack((Obs[jstars,0],Obs[jstars,1],1.))
    xg   = ct.EquatorialToGalactic(xe) 
    obsl = xg[0,0] # rad
    obsb = xg[0,1] # rad
    
    # Coordinates of star
    # obsStar  - varpi,Jmag,Hmag,Kmag
    # eobsStar - evarpi,eJmag,eHmag,eKmag        
    obsStar   = np.hstack((Obs[jstars,2:6]))
    eobsStar  = np.hstack((eObs[jstars,2:6]))
    obsvarpi  = obsStar[0]
    eobsvarpi = eobsStar[0]
              
    # Calculate grid of posterior probabilities
    nsig         = 3.
    ns           = 1000
    varpimin     = obsvarpi-nsig*eobsvarpi
    if (varpimin<0.):
        varpimin = 0.001
    varpimax     = obsvarpi+nsig*eobsvarpi
    varpigrid    = np.linspace(varpimin,varpimax,ns)
    sgrid        = 1./varpigrid
    sgrid        = np.sort(sgrid)
    spost,dmpost,ppost = \
        ad.pPostGrid(obsStar,eobsStar,obsl,obsb,sgrid,usegalprior)

    # Calculate marginalized posterior probabilities and samples
    timestart = time.time()
    dmsamp    = ad.calcMargAgeDist(spost,dmpost,ppost,nsamp)
    dmmoments[jstars,:] = np.array([np.mean(dmsamp),np.std(dmsamp)])
    print("Mean and error in distance modulus: "+str(np.round(dmmoments[jstars,0],3))+"+/"+str(np.round(dmmoments[jstars,1],3)))
    print("Distance modulus from parallax: "+str(5.*np.log10(1./obsvarpi * 100.)))
    timeend   = time.time()
    timetaken = timeend-timestart
    print("Run took "+ str(timetaken) + "s.")
    print(" ")        

#%% Save posterior samples to files
dmmomentsfile = "../results/apogee/dist/dmmoments_apogeetgas_distonly"
if (usegalprior==True):
    fileinfo = "_usegalprior"
else:
    fileinfo = "_nogalprior"
dmmomentsfile += fileinfo+".txt"
print(dmmomentsfile)
#%%
np.savetxt(dmmomentsfile,dmmoments)

#%% CREATE PLOTS

dmmoments = np.loadtxt(dmmomentsfile)

dmmin = 5.
dmmax = 25.

# Compare model/parallax distance modulus
dmvarpi = 5.*np.log10(100./Obs[:,2])
plt.rc('font',family='serif')
fig0 = plt.figure(figsize=(5,5))        
plt.scatter(dmvarpi,dmmoments[:,0],s=30, facecolors='none', edgecolors='r')
plt.plot([dmmin,dmmax],[dmmin,dmmax],':k')
plt.title(fileinfo,fontsize=16)
plt.xlim(dmmin,dmmax)
plt.ylim(dmmin,dmmax)
plt.xlabel("DM (mag)", fontsize=14)
plt.ylabel(r"DM$_{\mathrm{BNN}}$ (kpc)", fontsize=14)
fig0.savefig("../plots/apogee/dist/apogee_astrom_dmcomp"+fileinfo+".eps",fmt="eps")