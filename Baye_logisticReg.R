###############################
# Author:
# Marco A. Aquino-López
# Code:
# Bayesian Logistic Regression
# with measurement error
# for early Irish
###############################

rm(list=ls())   # Clear the workspace

library(R2jags)
library(boot)   # Package contains the logit transform
library(DMwR)   # Package contaning the unscale function
library(dummies)# Package containg the dummy.data.frame() function
library(beepr)  # Package containing the beep() function
library(HDInterval)  #Package containng the hdi() function
#set a seed
set.seed(12345678)
#set plot x interva;
x.int <-  765:880
#iterations and burnin
iter     = 7e5
burn     = iter-7e4
thinning = 125

# Read data
etar <- read.csv("~/github/ChronHib_B-logit/Data/etar/etar.csv")
N <- length(etar$Year_Upper)
etar$Year_Lower <- eta$Year_Lower
etar$Year_Upper <- etar$Year_Upper
etar$mean.age.us <- etar$Year_Lower+ (etar$Year_Upper-etar$Year_Lower )/2
scaled <- scale(etar$mean.age.us)
etar$Year_Lower <- scale(etar$Year_Lower,attr(scaled, 'scaled:center'),attr(scaled, 'scaled:scale'))
etar$Year_Upper <- scale(etar$Year_Upper,attr(scaled, 'scaled:center'),attr(scaled, 'scaled:scale'))
etar$sd <- as.numeric(abs(etar$Year_Lower-etar$Year_Upper))/6
etar$mean.age <- as.numeric(scaled)
etar$Context <- as.character(etar$Context)
dummies <- dummy(etar$Context)

# Jags code to fit the model to the simulated data
model_code = '
model
{
  # Likelihood
  for (i in 1:N) {
    y[i] ~ dbin(p[i], K)
    logit(p[i]) <- alpha + beta * x_1[i] + alpha1 * dummies[i,1] + alpha2 * dummies[i,2] + x_1[i] * beta_c1 * dummies[i,1] + x_1[i] * beta_c2 * dummies[i,2]
    x_1[i] ~ dnorm(mu_1[i], sigma_1[i]^-2)
  }
  # Priors
  alpha ~ dnorm(0, 10^-2)
  alpha1 ~ dnorm(0, 10^-2)
  alpha2 ~ dnorm(0, 10^-2)
  beta ~ dnorm(0, 10^-2)
  beta_c1 ~ dnorm(0, 10^-2)
  beta_c2 ~ dnorm(0, 10^-2)
}
'

# Set up the data
model_data = list(N = N, y = etar$Variant..0...e..1...i., mu_1 = etar$mean.age, sigma_1 = etar$sd,
                  dummies=dummies,
                  K = 1)

# Choose the parameters to watch
model_parameters =  c('alpha','beta',"alpha1","alpha2"," beta_c1","beta_c2","x_1")

# Run the model
 model_run = jags(data = model_data,n.iter = iter,n.burnin = burn,
                  parameters.to.save = model_parameters,
                  model.file = textConnection(model_code),n.chains = 3)

# save(file="~/OneDrive - Maynooth University/Elliott/e vs i/jags_modrun.Rdata", list="model_run")

load("~/OneDrive - Maynooth University/Elliott/e vs i/jags_modrun.Rdata")
recompile(object = model_run,n.iter = 1)
mod <- as.mcmc(model_run,thin=thinning)
# Simulated results -------------------------------------------------------

# Check the output - are the true values inside the 95% CI?
# Also look at the R-hat values - they need to be close to 1 if convergence has been achieved
plot(model_run)
print(model_run)


post = print(model_run)

#dates
x_1_mean = unscale(post$mean$x_1,scaled)
x_1_sd = post$sd$x_1*attr(x = scaled, which = "scaled:scale")

#betas now
alpha = as.matrix(mod[,1,])

alpha1 =  as.matrix(mod[,2,])
alpha2 =  as.matrix(mod[,3,])
beta =  as.matrix(mod[,4,])
beta_c1 =  as.matrix(mod[,5,])
beta_c2 =  as.matrix(mod[,6,])

alpha_mean = post$mean$alpha[1]
beta_mean = post$mean$beta[1]
alpha1_mean = post$mean$alpha1[1]
alpha2_mean = post$mean$alpha2[1]
beta_c1_mean = post$mean$beta_c1[1]
beta_c2_mean = post$mean$beta_c2[1]

alpha_sd = post$sd$alpha[1]
beta_sd = post$sd$beta[1]
alpha1_sd = post$sd$alpha1[1]
alpha2_sd = post$sd$alpha2[1]
beta_c1_sd = post$sd$beta_c1[1]
beta_c2_sd = post$sd$beta_c2[1]

# As we have two explanatory variables I'm going to create two plots
# holding one of the variables fixed whilst varying the other
prob <- function(x,C1=1,low=F,inter=T){
  if(C1==1){C2=0}else{if(C1==0){C2=1}else{C1=0;C2=0}}
  p1 <- inv.logit(alpha + beta*x + C1*(alpha1 + beta_c1 * x) + C2 * (alpha2 + beta_c2 * x) )
  int <- hdi(p1)
  if (inter==T){
    return(int[low+1])
  }else{
    return(p1)
  }
}

par(mfrow=c(1,1),mar=c(4.1,4.2,1.1,1.1))
plt.xs <- seq(from=min(scaled),to = max(scaled),length.out = length(x.int))

plot(x_1_mean, etar$Variant,pch='|',ylab = 'Variation Proability',xlab = 'Age',
     col=rgb(0,0,0,.1),xlim = c(760,880))

lines(x.int,
      inv.logit(alpha_mean + beta_mean*plt.xs + alpha1_mean + beta_c1_mean * plt.xs),
      col = rgb(1,0,0,.9),pch=16,cex=.9,lty = 2)
lines(x.int,
      inv.logit(alpha_mean + beta_mean*plt.xs + alpha2_mean + beta_c2_mean * plt.xs),
      col = rgb(0,0,1,.9),pch=16,cex=.9,lty = 2)
lines(x.int,
      inv.logit( alpha_mean + beta_mean * plt.xs),
      col = rgb(0,1,0,.9),pch=16,cex=.9,lty = 2)

lowint=NULL;uppint=NULL
for (k in plt.xs){
  lowint=c(lowint,prob(k,C1=1))
  uppint=c(uppint,prob(k,C1=1,low = T))
}
lines(x.int,lowint,col = rgb(1,0,0,.9),pch=16,cex=.9,lty = 2)
lines(x.int,uppint,col = rgb(1,0,0,.9),pch=16,cex=.9,lty = 2)

lowint=NULL;uppint=NULL
for (k in plt.xs){
  lowint=c(lowint,prob(k,C1=0))
  uppint=c(uppint,prob(k,C1=0,low = T))
}
lines(x.int,lowint,col = rgb(0,0,1,.9),pch=16,cex=.9,lty = 2)
lines(x.int,uppint,col = rgb(0,0,1,.9),pch=16,cex=.9,lty = 2)


lowint=NULL;uppint=NULL
for (k in plt.xs){
  lowint=c(lowint,prob(k,C1=-1))
  uppint=c(uppint,prob(k,C1=-1,low = T))
}
lines(x.int,lowint,col = rgb(0,1,0,.9),pch=16,cex=.9,lty = 2)
lines(x.int,uppint,col = rgb(0,1,0,.9),pch=16,cex=.9,lty = 2)

######## Context
for (i in 1:length(alpha)){
  lines(x.int,
        inv.logit(alpha[i] + beta[i]*plt.xs + alpha1[i] + beta_c1[i] * plt.xs),
        col = rgb(1,0,0,.01),pch=16,cex=.9)
  lines(x.int,
        inv.logit(alpha[i] + beta[i]*plt.xs + alpha2[i] + beta_c2[i] * plt.xs),
        col = rgb(0,0,1,.01),pch=16,cex=.7)
  lines(x.int,
        inv.logit( alpha[i] + beta[i] * plt.xs),
        col = rgb(0,1,0,.01),pch=16,cex=.7)
}


beep()

