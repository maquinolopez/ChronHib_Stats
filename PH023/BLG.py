###################################################
# Bayesian Logistic Regression 
# with measurement error in Early Irish
#
# Author:
# Marco A. Aquino-López
###################################################
import pytwalk
import numpy as np
import scipy as sp
import pandas as pd
from scipy.special import logit, expit
import random
import os
from sklearn.preprocessing import scale, StandardScaler,normalize
# Set seed
random.seed( 12345 )

# Read data
dat     = pd.read_csv (os.path.expanduser("~")+"/github/ChronHib_B-logit/PH023/PH023.csv") 
T_id    = pd.read_csv (os.path.expanduser("~")+"/github/ChronHib_B-logit/TextID.csv")

# Define parameters of MCMC
it = 1000

# Define prior ages
dat["m_ag"] = "" 
dat["sd_a"] = ""
dat["lsup"] = ""
dat["linf"] = ""
IDS = np.array(T_id["Text_ID"])
Tid = np.array(T_id[['mean','sd','UPPER','LOWER']])
datnew = np.array(dat[["m_ag",'sd_a',"lsup","linf"]])


for i in range(dat.shape[0]):
    i_d	= dat["TextID"][i].item()
    k   = np.where( IDS == str(i_d) )[0].item()
    datnew[i,0] = Tid[k,0]
    datnew[i,1] = Tid[k,1]  
    datnew[i,2] = Tid[k,2]  
    datnew[i,3] = Tid[k,3] 


dat[["m_ag",'sd_a',"lsup","linf"]] = datnew
datnew = datnew.astype(float)
# scale ages for faster convergence
scl_center  = datnew.mean(axis=0)[0]
scl_scale   = datnew.std(axis=0)[0]
scl_dat     = (datnew-scl_center)/scl_scale
#  
Variants    = dat["Variant"]-1
scl_dat[:,1]= scl_dat[:,1]+(scl_center/scl_scale) 
sddates     = scl_dat[:,1]
# Define functions
aloc	= 0
bloc	= 1
sdab	= 10**-1

def loglike(param):
        dates 	= param[2:]
        alpha 	= param[aloc]
        beta  	= param[bloc]
        p       = expit(alpha + beta*dates)
        like	= Variants * np.log(p) + (1 - Variants) * np.log(1-p)
        like 	= like - np.array(scl_dat[:,1] * (dates-scl_dat[:,0])**2).sum()
        return(like)	


def logprior(param):
    beta    = param[bloc]
    alpha   = param[aloc]
    palpha  = -(.5 / sdab) * alpha**2 
    pbeta    = -(.5 / sdab) * beta**2
    prior   = palpha + pbeta
    return(prior)



def obj(param):
    ob  = -np.array(logprior(param),loglike(param)).sum()
    return(ob)


def support(param):
    dates   = param[2:]   
    alpha   = param[aloc]
    beta    = param[bloc]
    limsup  = np.array(dates > scl_dat[:,2]).sum()
    liminf  = np.array(dates < scl_dat[:,3]).sum()
    sup     = np.array(limsup,liminf).sum()
    if sup==0:
        return(True)
    else:
        return(False)

# Define and run MCMC
ini1	= np.random.normal(np.append([0,0],scl_dat[:,0]), np.append([sdab,sdab],scl_dat[:,1]/15) )
support(ini1)
ini2    = np.random.normal(np.append([0,0],scl_dat[:,0]), np.append([sdab,sdab],scl_dat[:,1]/15) )
support(ini2)


scl_dat[:,1]    = .5*(scl_dat[:,1]+(scl_center/scl_scale) )**-2

#get burnin
twalk = pytwalk.pytwalk(n=len(ini1), U=obj, Supp=support,ww=[ 0.0, 0.4918, 0.4918, 0.0082+0.082, 0.0])   #
twalk.Run(T=1000*len(ini1),x0=ini1,xp0=ini2)






