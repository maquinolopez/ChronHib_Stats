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
import matplotlib as mp
import pandas as pd
from scipy.special import logit, expit
import random
import os
from sklearn.preprocessing import scale, StandardScaler,normalize
# Set seed and set if MCMC is to be run
#random.seed( 12345 )
it = 1000
Run     = True

# Read data
dat     = pd.read_csv (os.path.expanduser("~")+"/github/ChronHib_B-logit/PH023/PH023.csv") 
T_id    = pd.read_csv (os.path.expanduser("~")+"/github/ChronHib_B-logit/TextID.csv")

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
    p           = expit(alpha + beta*dates)
    like	= Variants * np.log(p) + (1 - Variants) * np.log(1-p)
    like 	= like - np.array(scl_dat[:,1] * (dates-scl_dat[:,0])**2).sum()
    return(like)	


def logprior(param):
    beta    = param[bloc]
    alpha   = param[aloc]
    palpha  = -(.5 / sdab) * alpha**2 
    pbeta   = -(.5 / sdab) * beta**2
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
    sup     = sum([limsup,liminf])
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

###### Ini MCMC
n = len(ini1)
burnin, thin = 200000, 100

#get burnin
if Run==True :
    twalk   = pytwalk.pytwalk(n=len(ini1), U=obj, Supp=support,ww=[ 0.0, 0.4918, 0.4918, 0.0082+0.082, 0.0])   #

    output  = np.array([])
    energy  = np.array([])
    i, k, k0, n, x0, xp0, U, Up   = 0, 0, 0, len(ini1), ini1, ini2, obj(ini1), obj(ini2)

    while i < it:
        onemove = twalk.onemove( x0, obj(x0), xp0, obj(xp0) )
        k += 1
        if (sp.stats.uniform.rvs() < onemove[3]):
            x0, xp0, ke, A, U, Up = onemove
            k0 += 1
            if all([k % thin == 0, k > int(burnin)]):   #This indicates the burnin
                output = np.append(output,x0.copy() )
                energy = np.append(energy,U)
                if any([i % 10 == 0, i == 0]):
                    print('{}%'.format(int(100*(i+.0)/it)) )
                i += 1
        else:
            if all([k % thin == 0, k > int(burnin)]):
                output = np.append(output,x0.copy() )
                energy = np.append(energy,U)
                if any([i % 10 == 0, i == 0]):
                    print('{}%'.format(int(100*(i+.0)/it)) )
                i += 1
    output = np.reshape(output,(it,len(ini1)))
    np.savetxt(os.path.expanduser("~")+"/github/ChronHib_B-logit/PH023/output.csv", output, delimiter=',',fmt='%1.10f')
    np.savetxt(os.path.expanduser("~")+"/github/ChronHib_B-logit/PH023/energy.csv", energy, delimiter=',',fmt='%1.10f')
else:
    output = np.genfromtxt(os.path.expanduser("~")+"/github/ChronHib_B-logit/PH023/output.csv", delimiter=",")
    energy = np.genfromtxt(os.path.expanduser("~")+"/github/ChronHib_B-logit/PH023/energy.csv", delimiter=",")
    print(output)

#print('{}'.format(output) )
#print('{}'.format(energy))


fig = mp.pyplot.figure(figsize=(6, 6))
grid        = mp.pyplot.GridSpec(6, 6, hspace=1, wspace=1.)
lit         = fig.add_subplot(grid[ 2: ,: ])
Energy      = fig.add_subplot(grid[0:2,0:2], xticklabels=[])
alpha       = fig.add_subplot(grid[0:2,2:4], xticklabels=[]) #Cambiar a y cuando se pase a la densidad
beta        = fig.add_subplot(grid[0:2,4: ], xticklabels=[])

Energy.plot(-energy,c='gray',lw=.5,alpha=.8)
Energy.set_title('Log of objective',size=11)

alpha.set_title('alpha',size=11)
kr_mem      = sp.stats.gaussian_kde(dataset=output[:,0])
x           = np.linspace(min(output[:,0].flatten()),max(output[:,0].flatten()),150)
#alpha.plot(x, kr_mem.evaluate(x), linestyle='solid', c='gray', lw=1,alpha=.8)
alpha.plot(output[:,0].flatten(), linestyle='solid', c='gray', lw=1,alpha=.8)


beta.set_title('beta',size=11)
kr_mem      = sp.stats.gaussian_kde(dataset=output[:,1])
x           = np.linspace(min(output[:,1].flatten()),max(output[:,1].flatten()),150)
#beta.plot(x, kr_mem.evaluate(x), linestyle='solid', c='gray', lw=1,alpha=.8)
beta.plot(output[:,1].flatten(), linestyle='solid', c='gray', lw=1,alpha=.8)

x           = np.linspace(min(datnew[:,0])-50,max(datnew[:,0])+50,150)
xus         = (x-scl_center)/scl_scale
print(x)

for param in output[:,0:2]:
    lit.plot(x,expit(param[0] + param[1]*xus),color='black',alpha = .02)



mp.pyplot.savefig(os.path.expanduser("~")+"/github/ChronHib_B-logit/PH023/PH023fig.pdf",bbox_inches = 'tight')
#mp.pyplot.show(fig)


