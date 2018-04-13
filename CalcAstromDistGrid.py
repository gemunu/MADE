"""
BAYESIAN METHOD TO CALCULATE DISTANCES AND ABSOLUTE J-BAND MAGNITUDES FOR 
ASTROMETRIC DATA (PHOTOMETRIC DATA USED FOR DUST CALCULATION)
"""
import numpy as np
import CoordTrans as ct
import dill as dill
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import AstroMethods as am
import mwdust

class AstromDist:
   
   ## CLASS CONSTRUCTOR
    # solpos        - solar position                                         [array]
    # dust          - whether to calculate extinction using Jo Bovy's mwdust [double]
    def  __init__(self,solpos,dust):
        self.solpos = np.copy(solpos)
        self.dust   = np.copy(dust)
        
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
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thin disc df
    def bulgedf(self,R,z):
        
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
                   
        # Position df
        q     = 0.5
        gamma = 0.0
        delta = 1.8
        r0    = 0.075
        rt    = 2.1
        m     = np.sqrt((R/r0)**2.+(z/(q*r0))**2.)
        pr    = ((1+m)**(gamma-delta))/(m**gamma) * np.exp(-(m*r0/rt)**2.)
        
        pbulge = pr
            
        return(pbulge)
           
    ## THIN DISC MODEL
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thin disc df
    def thindiscdf(self,R,z):
        
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
                      
        # Position df
        Rd    = 2.6
        zd    = 0.3
        pr    = np.exp(-R/Rd-np.abs(z)/zd)
        
        return(pr)
        
    ## THICK DISC MODEL
    # z   - cylindrical polar z [vector]
    # Returns unnormalized thick disc df
    def thickdiscdf(self,R,z):  
        
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
        
        # Distance df
        Rd    = 3.6
        zd    = 0.9
        pr    = np.exp(-R/Rd-np.abs(z)/zd)
                
        return(pr)
        
    ## STELLAR HALO MODEL
    # R   - cylindrical polar R [vector]
    # z   - cylindrical polar z [vector]
    # Returns unnormalized stellar halo df    
    def stellarhalodf(self,R,z):
        
        R   = np.atleast_1d(R)
        z   = np.atleast_1d(z)
               
        # Distance df
        rs    = np.sqrt(R**2.+z**2.)
        pr    = rs**(-3.39)
            
        return(pr)
        
    ## CALCULATE WEIGHTS ON THICK DISC AND STELLAR HALO FOR MILKY WAY
    # Returns fthick, fhalo    
    def calcDfNorms(self):
        
        solpos = np.copy(self.solpos)
        
        # Solar position
        R0 = solpos[0]
        z0 = solpos[1]
        
        # Df values at solar position marginalizd over ages and metallicities
        bulgedf       = self.bulgedf(R0,z0)
        thindiscdf    = self.thindiscdf(R0,z0)            
        thickdiscdf   = self.thickdiscdf(R0,z0)        
        stellarhalodf = self.stellarhalodf(R0,z0)

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
        
    ## PRIOR PROBABILITY ON DISTANCE
    # mods     - model distances [vector]
    # obsl     - Galactic longitude (rad) of observed star
    # obsb     - Galactic latitude (rad) of observed star
    # galprior - whether to use galaxy prior
    def pPrior(self,mods,obsl,obsb,usegalprior):
        
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
          
            prior = self.fbulge*self.bulgedf(R,z)+self.thindiscdf(R,z)+\
                    self.fthick*self.thickdiscdf(R,z)+\
                    self.fhalo*self.stellarhalodf(R,z)
        else:
            prior = np.ones_like(mods)
               
        return(prior)
            
    ## POSTERIOR PROBABILITY GRID FOR MODEL DISTANCE AND ABSOLUTE J-BAND MAGNITUDE
    # obsStar     - varpi,Jmag,Hmag,Kmag of observed star
    # eobsStar    - evarpi,eJmag,eHmag,eKmag of observed star
    # obsl        - Galactic longitude (rad) of observed star
    # obsb        - Galactic latitude (rad) of observed star
    # mods        - model grid of distances
    # usegalprior - whether to use galaxy prior
    # Returns 1D grids for distance, intrinsic luminosity, and posterior probability
    def pPostGrid(self,obsStar,eobsStar,obsl,obsb,mods,usegalprior):
        
        # Observations
        varpi,appJ,appH,appK     = obsStar
        evarpi,eappJ,eappH,eappK = eobsStar
              
        # Length of distance grid
        ns = len(mods)
        
        # Estimate extinction corrections if desired for matrix of distances
        spost    = np.copy(mods)
        if (self.dust==True):
            appJ = appJ-self.dustmapJ(obsl/np.pi*180.,obsb/np.pi*180.,spost)
            appH = appH-self.dustmapH(obsl/np.pi*180.,obsb/np.pi*180.,spost)
            appK = appK-self.dustmapKs(obsl/np.pi*180.,obsb/np.pi*180.,spost)
        
        # Output grids
        absJpost = am.absMag(appJ,mods)
        dmpost   = appJ-absJpost
                
        # Calculate likelihood
        modvarpi = 1./mods
        plh      = np.exp(-(modvarpi-varpi)**2./(2.*evarpi**2.))/np.sqrt(2.*np.pi*evarpi**2.)                                                          
                                      
        # Calculate prior
        pprior  = self.pPrior(mods,obsl,obsb,usegalprior)
                              
        return(spost,dmpost,plh*pprior)
               
    ## MARGINALIZED POSTERIOR SAMPLES
    # spost        - 4D mass grid (nage*nmh*nmass*ns)
    # dmpost       - 4D mass grid (nage*nmh*nmass*ns)
    # ppost        - 4D posterior probability grid (nage*nmh*nmass*ns)
    # nsamp        - number of posterior samples
    # Returns marginalized posterior density functions and samples for age and distance
    def calcMargAgeDist(self,spost,dmpost,ppost,nsamp):
        
        # Distance modulus
        cdm = cumtrapz(ppost,dmpost,initial=0)
        cdm = cdm/np.max(cdm)
        dm_invcdf_interp = interp1d(cdm,dmpost)
        unifrand         = np.random.uniform(size=nsamp)
        dmsamp           = dm_invcdf_interp(unifrand)
        
        return(dmsamp)