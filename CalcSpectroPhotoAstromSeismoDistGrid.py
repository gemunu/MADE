"""
BAYESIAN METHOD TO CALCULATE MASSES, AGES, METALLICITIES AND DISTANCES FOR 
SPECTRO-PHOTO-ASTROM-SEISMO DATA
"""
import numpy as np
import CoordTrans as ct
import dill as dill
from scipy.integrate import dblquad,trapz,cumtrapz
from scipy.interpolate import interp1d
import AstroMethods as am
import DistFunc as df
import mwdust
import matplotlib.pyplot as plt

class SpectroPhotoAstromSeismoDist:
    
    ## CLASS CONSTRUCTOR
    # solpos        - solar position                                         [array]
    # dust          - whether to calculate extinction using Jo Bovy's mwdust [double]
    # taum          - age of the Universe
    def  __init__(self,solpos,dust,taum):
        self.solpos = np.copy(solpos)
        self.dust   = np.copy(dust)
        self.taum   = np.copy(taum)
        
        # Undill Parsec isochrones and interpolants
        print(" ")
        print("Undilling isochrones and interpolants...")
        with open("stellarprop_parsecdefault_currentmass.dill", "rb") as input:
            self.pi = dill.load(input)
        print("...done.")
        print(" ")

        self.isoage = np.copy(self.pi.isoage) 
        self.isomh  = np.copy(self.pi.isomh)
        
        # Calculate galaxy model normalizations
        fbulge,fthick,fhalo = self.calcDfNorms()
        self.fbulge  = np.copy(fbulge)
        self.fthick  = np.copy(fthick)
        self.fhalo   = np.copy(fhalo)
        
        # Initialize Sale, 2014 dust map if desired and share with class
        if (dust==True):
            self.dustmapJ  = mwdust.Combined15(filter = '2MASS J')
            self.dustmapH  = mwdust.Combined15(filter = '2MASS H')
            self.dustmapKs = mwdust.Combined15(filter = '2MASS Ks')
            
    ## BULGE MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thin disc df
    def bulgedf(self,tau,mh,R,z):
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = -0.3       
        sigmh = 0.3
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df
        mutau  = 5.
        sigtau = 5.
        ptau  = np.exp(-(tau-mutau)**2./(2.*sigtau**2.))/np.sqrt(2.*np.pi*sigtau**2.) 
            
        # Position df
        q     = 0.5
        gamma = 0.0
        delta = 1.8
        r0    = 0.075
        rt    = 2.1
        m     = np.sqrt((R/r0)**2.+(z/(q*r0))**2.)
        pr    = ((1+m)**(gamma-delta))/(m**gamma) * np.exp(-(m*r0/rt)**2.)
        
        pbulge = pmh*ptau*pr
            
        return(pbulge)
           
    ## THIN DISC MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thin disc df
    def thindiscdf(self,tau,mh,R,z):
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = 0.0       
        sigmh = 0.2
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df
        tauF  = 8.
        taus  = 0.43
        index = tau <=10.
        ptau  = np.zeros_like(tau)
        ptau[index] = np.exp(tau[index]/tauF - taus/(self.taum-tau[index]))
            
        # Position df
        Rd    = 2.6
        zd    = 0.3
        pr    = np.exp(-R/Rd-np.abs(z)/zd)
        
        pthin = pmh*ptau*pr
            
        return(pthin)
        
    ## THICK DISC MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thick disc df
    def thickdiscdf(self,tau,mh,R,z):  
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = -0.6       
        sigmh = 0.5
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df        
        tauF  = 8.
        taus  = 0.43
        index = tau>10.
        ptau  = np.zeros_like(tau)
        ptau[index] = np.exp(tau[index]/tauF - taus/(self.taum-tau[index]))
            
        # Distance df
        Rd    = 3.6
        zd    = 0.9
        pr    = np.exp(-R/Rd-np.abs(z)/zd)
        
        pthick = pmh*ptau*pr
        
        return(pthick)
        
    ## STELLAR HALO MODEL
    # tau - age                 [vector]
    # mh  - metallicity         [vector]
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized stellar halo df    
    def stellarhalodf(self,tau,mh,R,z):
        
        tau = np.atleast_1d(tau)
        mh  = np.atleast_1d(mh)
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Metallicity df
        mumh  = -1.6     
        sigmh = 0.5
        pmh   = np.exp(-(mh-mumh)**2./(2.*sigmh**2.))/np.sqrt(2.*np.pi*sigmh**2.) 
        
        # Age df        
        mutau  = 11.0
        sigtau = 1.0
        ptau  = np.exp(-(tau-mutau)**2./(2.*sigtau**2.))/np.sqrt(2.*np.pi*sigtau**2.) 
       
        # Distance df
        rs    = np.sqrt(R**2.+z**2.)
        pr    = rs**(-3.39)
        
        phalo = pmh*ptau*pr
            
        return(phalo)
        
    ## CALCULATE WEIGHTS ON THICK DISC AND STELLAR HALO FOR MILKY WAY
    # Returns fthick, fhalo    
    def calcDfNorms(self):
        
        solpos = np.copy(self.solpos)
        
        # Solar position
        R0 = solpos[0]
        z0 = solpos[1]
        
        # Df values at solar position marginalizd over ages and metallicities
        mhmin  = np.min(self.isomh)
        mhmax  = np.max(self.isomh)
        agemin = np.min(self.isoage)
        bulgedf = dblquad(self.bulgedf,
                          mhmin,mhmax,
                          lambda tau: agemin, lambda tau: self.taum,
                          args=(R0,z0))[0]
        thindiscdf    = dblquad(self.thindiscdf,
                                mhmin,mhmax,
                                lambda tau: agemin, lambda tau: self.taum,
                                args=(R0,z0))[0]            
        thickdiscdf   = dblquad(self.thickdiscdf,
                                mhmin,mhmax,
                                lambda tau: agemin, lambda tau: self.taum,
                                args=(R0,z0))[0]            
        stellarhalodf = dblquad(self.stellarhalodf,
                                mhmin,mhmax,
                                lambda tau: agemin, lambda tau: self.taum,
                                args=(R0,z0))[0]

        # Calcualte weights on galaxy components
        print("Calculating weights on galaxy components...")
        rbulgethin = 0.001        
        rthickthin = 0.15
        rhalothin  = 0.005
        fbulge     = rbulgethin*thindiscdf/bulgedf
        fthick     = rthickthin*thindiscdf/thickdiscdf
        fhalo      = rhalothin*thindiscdf/stellarhalodf
        print("     fbulge = "+str(np.round(fbulge,3)))
        print("     fthick = "+str(np.round(fthick,3)))
        print("     fhalo  = "+str(np.round(fhalo,3)))
        print("...done.")

        return(fbulge,fthick,fhalo)
        
    ## PRIOR PROBABILITY ON tau, mh, mass, s
    # modtau  - model ages [vector]
    # modmh   - model metallicities [vector]
    # modmass - model masses [vector]
    # mods    - model distances [vector]
    # obsl    - Galactic longitude (rad) of observed star
    # obsb    - Galactic latitude (rad) of observed star
    # galprior - whether to use galaxy prior
    def pPrior(self,modtau,modmh,modmass,mods,obsl,obsb,usegalprior):
        
        # Copy shared variables
        solpos  = np.copy(self.solpos)
                
        # Define prior
        if (usegalprior==True):
            shape = np.shape(mods)
            R     = np.zeros_like(mods)
            z     = np.zeros_like(mods)
            if (len(shape)==1):
                xg = np.column_stack([obsl+mods*0.,obsb+mods*0.,mods])
                xp = ct.GalacticToPolar(xg,solpos)
                R  = xp[:,0]
                z  = xp[:,2]
            if (len(shape)==2):
                for j in range(shape[0]):
                    xg     = np.column_stack([obsl+mods[j,:]*0.,obsb+mods[j,:]*0.,mods[j,:]])
                    xp     = ct.GalacticToPolar(xg,solpos)
                    R[j,:] = xp[:,0]
                    z[j,:] = xp[:,2]
          
            prior = df.imf(modmass)*(self.fbulge*self.bulgedf(modtau,modmh,R,z)+\
                self.thindiscdf(modtau,modmh,R,z)+\
                self.fthick*self.thickdiscdf(modtau,modmh,R,z)+\
                self.fhalo*self.stellarhalodf(modtau,modmh,R,z))
        else:
            prior = np.ones_like(modmass)
               
        # Set prior to zero where parameters outside desired ranges (no need for mass)
        mhmin  = np.min(self.isomh)
        mhmax  = np.max(self.isomh)
        agemin = np.min(self.isoage)
        index = (modtau > self.taum) & (modtau < agemin) & \
                (modmh > mhmax) & (modmh < mhmin)
        prior[index] = 0.
    
        return(prior)
            
    ## POSTERIOR PROBABILITY GRID FOR MODEL tau, mh, mass, s
    # obsStar     - varpi,Jmag,Hmag,Kmag,teff,logg,mh,mass of observed star
    # eobsStar    - evarpi,eJmag,eHmag,eKmag,eteff,elogg,emh,emass of observed star
    # obsl        - Galactic longitude (rad) of observed star
    # obsb        - Galactic latitude (rad) of observed star
    # nsig        - how many sigmas to extend metallicity grid over
    # nmass       - number of masses in grid
    # modsgrid    - model grid of distances
    # usegalprior - whether to use galaxy prior
    # usedata     - what data to use (in order: appmag, col,logg,teff,feh,mass,parallax)
    # silent      - whether to print stuff during emcee run
    # Returns 4D grids for age, metallicity, mass, distance, and posterior probability
    def pPostGrid(self,obsStar,eobsStar,obsl,obsb,nsig,nmass,modsgrid,
                  usegalprior,usedata,silent):
        
        # Observations
        varpi,appJ,appH,appK,teff,logg,mh,mass = obsStar
        evarpi,eappJ,eappH,eappK,eteff,elogg,emh,emass = eobsStar
        eJminK = np.sqrt(eappJ**2.+eappK**2.)
              
        # Construct grids
        nage       = len(self.isoage)
        mhmax      = mh+nsig*emh
        mhmin      = mh-nsig*emh
        jmhmin     = np.sum(self.isomh<mhmin)-1
        jmhmax     = np.sum(self.isomh<mhmax)
        if (jmhmin < 0):
            jmhmin= 0
        if (jmhmax == len(self.isomh)):
            jmhmax = len(self.isomh)-1
        modmhgrid  = self.isomh[jmhmin:jmhmax+1]
        nmh        = len(modmhgrid)
        ns         = len(modsgrid)
        modsmat    = np.vstack([modsgrid]*nmass)
        
        # Output matrices
        agepost      = np.zeros([nage,nmh,nmass,ns])
        mhpost       = np.zeros([nage,nmh,nmass,ns])
        masspost     = np.zeros([nage,nmh,nmass,ns])
        spost        = np.zeros([nage,nmh,nmass,ns])
        log10agepost = np.zeros([nage,nmh,nmass,ns])
        dmpost       = np.zeros([nage,nmh,nmass,ns])
        ppost        = np.zeros([nage,nmh,nmass,ns])

        # Estimate extinction corrections if desired for matrix of distances
        if (self.dust==True):
            appJmat = appJ-self.dustmapJ(obsl/np.pi*180.,obsb/np.pi*180.,modsmat)
            appHmat = appH-self.dustmapH(obsl/np.pi*180.,obsb/np.pi*180.,modsmat)
            appKmat = appK-self.dustmapKs(obsl/np.pi*180.,obsb/np.pi*180.,modsmat)
        else:
            appJmat = np.tile(appJ,[nmass,ns])
            appHmat = np.tile(appJ,[nmass,ns])
            appKmat = np.tile(appJ,[nmass,ns])

        # Create nmass x ns matrices for colour
        JminKmat = appJmat - appKmat
                                      
        for jage in range(nage):
            for jmh in range(nmh):
                modtaumat  = np.zeros([nmass,ns]) + self.isoage[jage]
                modmhmat   = np.zeros([nmass,ns]) + modmhgrid[jmh]
                interpname = "age"+str(np.round(self.isoage[jage],10))+"mh"+str(modmhgrid[jmh])
                if (interpname in self.pi.isointerpdict):
                    isochrone = self.pi.isointerpdict[interpname]  
                else:
                    interpname = "age"+str(np.round(self.isoage[jage],11))+"mh"+str(modmhgrid[jmh])
                    if (interpname in self.pi.isointerpdict):
                        isochrone = self.pi.isointerpdict[interpname]  
                    else:
                        interpname = "age"+str(np.round(self.isoage[jage],12))+"mh"+str(modmhgrid[jmh])
                        if (interpname in self.pi.isointerpdict):
                            isochrone = self.pi.isointerpdict[interpname]  
                        else:
                            interpname   = "age"+str(np.round(self.isoage[jage],13))+"mh"+str(modmhgrid[jmh])
                            if (interpname in self.pi.isointerpdict):
                                isochrone = self.pi.isointerpdict[interpname]  
                            else:
                                interpname = "age"+str(np.round(self.isoage[jage],14))+"mh"+str(modmhgrid[jmh])                                
                isochrone  = self.pi.isointerpdict[interpname]
                massmin    = self.pi.massmindict[interpname]*1.01
                massmax    = self.pi.massmaxdict[interpname]*0.99
                massgrid   = np.logspace(np.log10(massmin),np.log10(massmax),nmass)
                modmassmat = np.column_stack([massgrid]*ns)
                
                # Fill output arrays
                agepost[jage,jmh,:,:]      = modtaumat
                mhpost[jage,jmh,:,:]       = modmhmat
                masspost[jage,jmh,:,:]     = modmassmat
                spost[jage,jmh,:,:]        = modsmat
                log10agepost[jage,jmh,:,:] = np.log10(modtaumat)
                absJpost                   = am.absMag(appJmat,modsmat)
                dmpost[jage,jmh,:,:]       = appJmat-absJpost
                
                # Likelihood of metallicity observable
                pmh = np.exp(-(mh-modmhmat)**2./(2.*emh**2.))/np.sqrt(2.*np.pi*emh**2.)             
                stellarpropmod = isochrone(modmassmat)
                modteffmat     = 10.**(stellarpropmod[0,:,:])
                modloggmat     = stellarpropmod[1,:,:]
                modabsJmat     = stellarpropmod[2,:,:]
                modabsHmat     = stellarpropmod[3,:,:]
                modabsKmat     = stellarpropmod[4,:,:]
                    
                # Likelihood of Teff, logg, and mass observables
                pteffmat = \
                    np.exp(-(teff-modteffmat)**2./(2.*eteff**2.))/np.sqrt(2.*np.pi*eteff**2.)
                ploggmat = \
                    np.exp(-(logg-modloggmat)**2./(2.*elogg**2.))/np.sqrt(2.*np.pi*elogg**2.)
                pmassmat = \
                    np.exp(-(mass-modmassmat)**2./(2.*emass**2.))/np.sqrt(2.*np.pi*emass**2.)
                    
                # Calculate model apparent magnitudes
                modappHmat  = am.appMag(modabsHmat,modsmat)
                modJminKmat = modabsJmat-modabsKmat
                 
                # Appmag and colour likelihood        
                pappmagmat = \
                    np.exp(-(appHmat-modappHmat)**2./(2.*eappH**2.))/np.sqrt(2.*np.pi*eappH**2.)
                pcolmat = \
                    np.exp(-(JminKmat-modJminKmat)**2./(2.*eJminK**2.))/np.sqrt(2.*np.pi*eJminK**2.)
                                    
                # Parallax likelihood
                modvarpimat = 1./modsmat
                pparmat = \
                    np.exp(-(modvarpimat-varpi)**2./(2.*evarpi**2.))/np.sqrt(2.*np.pi*evarpi**2.)                                                          
                                      
                # Calculate prior
                pprior  = self.pPrior(modtaumat,modmhmat,modmassmat,modsmat,obsl,obsb,usegalprior)

                # Calculate posterior probability              
                prob = np.copy(pprior)
                if (usedata[0]==True): # appmag
                    prob *= pappmagmat
                if (usedata[1]==True): # col
                    prob *= pcolmat
                if (usedata[2]==True): # logg
                    prob *= ploggmat
                if (usedata[3]==True): # Teff
                    prob *= pteffmat
                if (usedata[4]==True): # feh
                    prob *= pmh
                if (usedata[5]==True): # mass
                    prob *= pmassmat
                if (usedata[6]==True): # parallax
                    prob *= pparmat
                    
                ppost[jage,jmh,:,:] = np.copy(prob)
                if (silent==False):                
                    print("modtau: "+str(modtaumat))
                    print("modmh: "+str(modmhmat))
                    print("modmass: "+str(modmassmat))
                    print("mods: "+str(modsmat))   
                    print("Posterior probability: "+str(ppost[jage,jmh,:,:]))
                              
        return(agepost,mhpost,masspost,spost,log10agepost,dmpost,ppost)
        
    ## MARGINALIZED POSTERIOR SAMPLES
    # agepost      - 4D mass grid (nage*nmh*nmass*ns)
    # mhpost       - 4D mass grid (nage*nmh*nmass*ns)
    # masspost     - 4D mass grid (nage*nmh*nmass*ns)
    # spost        - 4D mass grid (nage*nmh*nmass*ns)
    # log10agepost - 4D mass grid (nage*nmh*nmass*ns)
    # dmpost       - 4D mass grid (nage*nmh*nmass*ns)
    # ppost        - 4D posterior probability grid (nage*nmh*nmass*ns)
    # nsamp        - number of posterior samples
    # Returns marginalized posterior density functions and samples for age and distance
    def calcMargAgeDist(self,agepost,mhpost,masspost,spost,log10agepost,dmpost,
                        ppost,nsamp):
        #
        nage,nmh,nmass,ns = np.shape(agepost)
        log10age  = log10agepost[:,0,0,0]
        dm        = dmpost[0,0,0,:]
        
        # Posterior probability marginalized over metallicity and mass
        ppostmargmhmass = np.zeros([nage,ns])
        for jage in range(nage):
            for js in range(ns):
                ppostmargmass = np.zeros(nmh)
                # First over mass
                for jmh in range(nmh):
                    ppostmargmass[jmh] = trapz(ppost[jage,jmh,:,js],masspost[jage,jmh,:,js])
                # Then over metallicity
                ppostmargmhmass[jage,js] = trapz(ppostmargmass,mhpost[jage,:,0,js])
                
        """        
        # Plot 2D posterior probability
        plt.rc('font',family='serif')  
        plt.figure(figsize=(5,5))    
        plt.contourf(age,dist,np.transpose(ppostmargmhmass)) 
        plt.title(r"$P(s,\tau|\mathrm{data})$")
        plt.xlabel(r"$\tau$ (Gyr)",fontsize=14)
        plt.ylabel(r"$s$ (kpc)",fontsize=14)
        """
        
        # Marginalize further over distance
        page = np.zeros(nage)
        for jage in range(nage):
            page[jage] = trapz(ppostmargmhmass[jage,:],spost[jage,0,0,:])

        # Marginalize further over age
        pdist = np.zeros(ns)
        for js in range(ns):
            pdist[js] = trapz(ppostmargmhmass[:,js],agepost[:,0,0,js])
                
        # log10age  
        clog10age = cumtrapz(page,log10age,initial=0)
        clog10age = clog10age/np.max(clog10age)
        
        # Distance modulus
        cdm = cumtrapz(pdist,dm,initial=0)
        cdm = cdm/np.max(cdm)
        
        # Generate posterior log10age samples
        log10age_invcdf_interp = interp1d(clog10age,log10age)
        unifrand               = np.random.uniform(size=nsamp)
        log10agesamp           = log10age_invcdf_interp(unifrand)
        
        # Generate posterior dm samples
        dm_invcdf_interp = interp1d(cdm,dm)
        unifrand         = np.random.uniform(size=nsamp)
        dmsamp           = dm_invcdf_interp(unifrand)
        
        return(log10agesamp,dmsamp)
        
"""
#%% TEST

### LOOK AT APOGEE-TGAS DATA TO GET AN IDEA OF ERRORS
datafile  = "../data/apogee_and_crossmatches/ApogeeCannonTgas.csv"
 
# Read data file (location_id,ra,dec,vlos,evlos,j,ej,h,eh,k,ek,teff,teff_err,
#                 logg,logg_err,m_h,m_h_err,alpha_m,alpha_m_err,ra_gaia,
#                 dec_gaia,ra_error,dec_error,parallax,parallax_error,
#                 pmra,pmra_error,pmdec,pmdec_error,meandist,diststd,lnm,
#                 elnm,lnage,elnage,mass,mass68L,mass68U,mass95L,mass95L,
#                 age,age68L,age68U,age95L,age95L)

data   = np.loadtxt(datafile,delimiter=',') 
        
# TGAS Ra, TGAS Dec, TGAS parallax, APOGEE vlos, TGAS PMRa, TGAS PMDec, 
# APOGEE J, APOGEE H, APOGEE K, APOGEE Te, APOGEE logg, APOGEE [M/H], 
# APOGEE [a/M], StarHorse s, Cannon ln(mass), Cannon ln(age)                    
Obs  = np.column_stack((data[:,19],data[:,20],data[:,23],data[:,3],data[:,25],
                        data[:,27],data[:,5],data[:,7],data[:,9],data[:,11],
                        data[:,13],data[:,15],data[:,17],data[:,29],data[:,31],
                        data[:,33]))
eObs = np.column_stack((data[:,21],data[:,22],data[:,24],data[:,4],data[:,26],
                        data[:,28],data[:,6],data[:,8],data[:,10],data[:,12],
                        data[:,14],data[:,16],data[:,18],data[:,30],data[:,32],
                        data[:,34]))
error               = eObs[:,2]/np.abs(Obs[:,2])
index               = ~np.isinf(error) & (Obs[:,2]>-9999) & (eObs[:,2]>-9999)
mean_parallax_error = np.mean(error[index])

error               = eObs[:,6]/np.abs(Obs[:,6])
index               = ~np.isinf(error) & (Obs[:,6]>-9999) & (eObs[:,6]>-9999)
mean_mag_error      = np.mean(error[index])

error               = eObs[:,9]/np.abs(Obs[:,9])
index               = ~np.isinf(error) & (Obs[:,9]>-9999) & (eObs[:,9]>-9999)
mean_te_error       = np.mean(error[index]) 
  
error               = eObs[:,10]/np.abs(Obs[:,10])
index               = ~np.isinf(error) & (Obs[:,10]>-9999) & (eObs[:,10]>-9999)
mean_logg_error     = np.mean(error[index])     

error               = eObs[:,11]
index               = ~np.isinf(error) & (Obs[:,11]>-9999) & (eObs[:,11]>-9999)
mean_mh_error       = np.mean(error[index])             

error               = eObs[:,14]/np.abs(Obs[:,14])
index               = ~np.isinf(error) & (Obs[:,14]>-9999) & (eObs[:,14]>-9999)
mean_lnm_error      = np.mean(error[index])         

error               = eObs[:,15]/np.abs(Obs[:,15])
index               = ~np.isinf(error) & (Obs[:,15]>-9999) & (eObs[:,15]>-9999)
mean_lnage_error    = np.mean(error[index])         

#%% INSTANTIATE CLASS   
solpos = np.array([8.3,0.014,-14.0,12.24+238.,7.25])
dust   = False
spas   = SpectroPhotoAstromSeismoDist(solpos,dust)

#%% TEST ALGORITHM WITH KNOWN AGE, METALLICITY, MASS, AND DISTANCE
agetest   = 8.0      # Gyr
lnagetest = np.log(agetest)
mhtest    = -0.1     # dex
masstest  = 0.8      # solar masses
lnmtest   = np.log(masstest)
stest     = 0.3      # kpc
varpitest = 1./stest # mas
 
# Find closest isochrone
mhdiff         = np.abs(spas.pi.isomh-mhtest)
mhiso          = spas.pi.isomh[mhdiff==np.min(mhdiff)][0]
agediff        = np.abs(spas.pi.isoage-agetest)
ageiso         = spas.pi.isoage[agediff==np.min(agediff)][0]
interpnamecurr = "age"+str(ageiso)+"mh"+str(mhiso)
isochrone      = spas.pi.isointerpdict[interpnamecurr]

# Predict stellar properties 
stellarproptest = isochrone(masstest)
tefftest        = 10.**(stellarproptest[0])
loggtest        = stellarproptest[1]
Jtest           = am.appMag(stellarproptest[2],stest)
Htest           = am.appMag(stellarproptest[3],stest)
Ktest           = am.appMag(stellarproptest[4],stest)

# obsStar  - varpi,Jmag,Hmag,Kmag,teff,logg,mh,lnm,lnage [array]
# eobsStar - evarpi,eJmag,eHmag,eKmag,eteff,elogg,emh,elnm,elnagetest [array]
obsStartest_exact = np.array([varpitest,Jtest,Htest,Ktest,tefftest,loggtest,mhtest,lnmtest,lnagetest])
sameerr = False
if (sameerr==True):
    eobsStartest = np.abs(obsStartest_exact)*0.1
else:
    eobsStartest      = np.zeros_like(obsStartest_exact)
    eobsStartest[0]   = mean_parallax_error/5. * np.abs(obsStartest_exact[0])
    eobsStartest[1:4] = mean_mag_error*np.abs(obsStartest_exact[1:4])
    eobsStartest[4]   = mean_te_error*np.abs(obsStartest_exact[4])/3.
    eobsStartest[5]   = mean_logg_error*np.abs(obsStartest_exact[5])/3.
    eobsStartest[6]   = mean_mh_error
    eobsStartest[7]   = mean_lnm_error*np.abs(obsStartest_exact[7])/3.
    eobsStartest[8]   = mean_lnage_error*np.abs(obsStartest_exact[8])
    
# Make realisation of obsStar
nobs = len(obsStartest_exact)
obsStartest = np.zeros(nobs)
for jobs in range(nobs):
    obsStartest[jobs] = np.random.normal(obsStartest_exact[jobs],eobsStartest[jobs])
            
#%% Calculate grid of posterior probabilities
nsig         = 3.
nmass        = 200
ns           = 100
varpimin     = obsStartest[0]-nsig*eobsStartest[0]
varpimax     = obsStartest[0]+nsig*eobsStartest[0]
varpigrid    = np.linspace(varpimin,varpimax,ns)
sgrid        = 1./varpigrid
sgrid        = np.sort(sgrid)
silent       = True
useparallax  = True
usemass      = True
usegalprior  = False
obsl         = 1.
obsb         = 1.
agepost,mhpost,masspost,spost,ppost = \
    spas.pPostGrid(obsStartest,eobsStartest,obsl,obsb,nsig,nmass,sgrid,useparallax,usemass,usegalprior,silent)

# Calculate marginalized posterior probabilities and samples
nsamp = 1000
age,page,cage,agesamp,dist,pdist,cdist,distsamp = \
    spas.calcMargAgeDist(agepost,mhpost,masspost,spost,ppost,nsamp)

#%% Create plots 
agemin = np.min(spas.isoage)
agemax = np.max(spas.isoage)
smin   = np.min(sgrid)
smax   = np.max(sgrid)

plt.rc('font',family='serif')  
fig0    = plt.figure(figsize=(8,8))    
fig0.subplots_adjust(hspace=0.4,wspace=0.3)

plt.subplot(3,2,1) 
plt.plot(age,page)
age_mod  = [agetest,agetest]
page_mod = [np.min(page),np.max(page)]
plt.plot(age_mod,page_mod,":k")
plt.xlim(agemin,agemax)
plt.xlabel(r"$\tau$ (Gyr)", fontsize=10)
plt.ylabel(r"$p_{\tau}$", fontsize=10)

plt.subplot(3,2,2) 
plt.plot(dist,pdist)
dist_mod = [stest,stest]
pdist_mod = [np.min(pdist),np.max(pdist)]
plt.plot(dist_mod,pdist_mod,":k")
plt.xlim(smin,smax)
plt.xlabel(r"$s$ (kpc)", fontsize=10)
plt.ylabel(r"$p_s$", fontsize=10)

plt.subplot(3,2,3) 
plt.plot(age,cage)
plt.xlim(agemin,agemax)
plt.xlabel(r"$\tau$ (Gyr)", fontsize=10)
plt.ylabel(r"$c_{\tau}$", fontsize=10)

plt.subplot(3,2,4) 
plt.plot(dist,cdist)
plt.xlim(smin,smax)
plt.xlabel(r"$s$ (kpc)", fontsize=10)
plt.ylabel(r"$c_s$", fontsize=10)

plt.subplot(3,2,5) 
plt.hist(agesamp,20)
plt.xlabel(r"$\tau$  (Gyr)", fontsize=10)

plt.subplot(3,2,6) 
plt.hist(distsamp,20)
plt.xlabel(r"$s$ (kpc)", fontsize=10)

fig0.savefig("../plots/apogee/dist/test.eps",format='eps',rasterized=True,bbox_inches='tight')
"""