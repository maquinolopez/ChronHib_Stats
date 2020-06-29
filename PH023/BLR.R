
###################################################
# Bayesian Logistic Regression 
# with measurement error in Early Irish
#
# Author:
# Marco A. Aquino-López
###################################################
rm(list = ls()) 	# clean R environment

# Load libraries
library(boot)   # Package containing the logit transform
library(DMwR)   # Package containing the unscale function
library(dummies)# Package containing the dummy.data.frame() function
library(beepr)  # Package containing the beep() function
library(HDInterval)  #Package containing the hdi() function
library(Rtwalk) # Package containing 

set.seed(123456) # set.seed
# Read data
dat <- read.csv('./PH023.csv');head(dat)
T_id <- read.csv('../TextID.csv');head(dat)

# Define parameters of MCMC
it = 500

# Define prior ages
dat$mean <- NA
dat$sd	 <- NA
dat$lsup <- NA
dat$linf <- NA
for (i in 1:dim(dat)[1]){
	i_d	<- dat[i,1]
	idloc	<- which(T_id$Text_ID==as.character(i_d))
	dat$mean[i] <- T_id$mean[idloc]
	dat$sd[i]   <- T_id$sd[idloc]
	dat$lsup[i] <- T_id$UPPER[idloc]
	dat$linf[i]   <- T_id$LOWER[idloc]
}
scaled	<- scale(dat$mean)			# scale ages for faster convergence
scl_center <- attr(scaled,"scaled:center")
scl_scale  <- attr(scaled,"scaled:scale")
dat$s_mean <- as.numeric(scaled)
dat$s_sd   <- .5/as.numeric(scale(dat$sd,scl_center,scl_scale) )^2
dat$s_lin  <- as.numeric(scale(dat$linf,scl_center,scl_scale) )
dat$s_lsp  <- as.numeric(scale(dat$lsup,scl_center,scl_scale) )
# Define functions
aloc	<- 1
bloc	<- 2
agloc	<- -c(1,2)
sdab	<- 10^-1

loglike <- function (param){
	dates 	<- param[agloc]
	alpha 	<- param[aloc]
	beta  	<- param[bloc]
	p <- inv.logit(alpha + beta*dates)
	like	<- dat$Variant * log(p) + (1 - dat$Variant) * log(1-p)
	like 	<- like - dat$s_sd * (dates-dat$s_mean)^2
	return(like)	
}

logprior<- function(param){
	#dates 	<- param[agloc]	alpha 	<- param[aloc]
	beta  	<- param[bloc]
	palpha	<- -.5 * sdab * alpha^2
	pbeta	<- -.5 * sdab * beta^2
	return(palpha + pbeta)


obj<- function(param){
	-sum(logprior(param),loglike(param)) 
}

support <- function(param){
	dates 	<- param[agloc]
	alpha 	<- param[aloc]
	beta  	<- param[bloc]
	limsup 	<- sum(dates > dat$s_lsp)
	liminf 	<- sum(dates < dat$s_lin)
	sup	<- sum(limsup,liminf)
	if (sup==0){return(TRUE)}
	else{return(FALSE)}
}

# Define and run MCMC

ini1	<- rnorm(dim(dat)[1]+2,c(0,0,dat$s_mean),c(3,3,dat$s_sd/5)) 
ini2	<- rnorm(dim(dat)[1]+2,c(0,0,dat$s_mean),c(3,3,dat$s_sd/5)) 

#get burnin
twalk <- Runtwalk(Tr = 8000 , Obj = obj,Supp = support,x0 = ini1,xp0 = ini2,PlotLogPost = FALSE)

#get iterations
Output  <- matrix(as.numeric(tail(twalk$output,1)),nrow=1 )
tm = 0
for (i in 1:500){
	x0    <- as.numeric(tail(twalk$output,1))
	xp0    <- as.numeric(tail(twalk$outputp,1))
	if (tm ==0 ){start.time <- Sys.time()}
	twalk <- Runtwalk(Tr = 1000, Obj = obj,Supp = support,x0 =x0 ,xp0 = xp0 ,PlotLogPost = FALSE)
	Output  <- rbind(Output ,as.numeric(tail(twalk$output,1)) )
	if (tm ==0 ){print(Sys.time()- start.time)}

}


plot(Output[,dim(Output)[2]])



