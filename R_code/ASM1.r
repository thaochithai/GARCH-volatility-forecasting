install.packages("nortest")
install.packages("ADGofTest")
library(xts)
library(zoo)
library(quantmod)
library(qrmtools)
library(qrmdata)
library(rugarch)
library(tseries)
library(moments)
library(ADGofTest) 

## Data
getSymbols("MSFT") 
msft <- MSFT['2012/2020']
msft.adj <- msft$MSFT.Adjusted
y <- returns(msft.adj)
y=na.omit(y) 
dim(y)
yw <- apply.weekly(y, FUN = colSums)
plot.zoo(y)
summary(y)
sd(y)
skewness(y)
kurtosis(y)
jarque.bera.test(y)

##unitroottest (stationary test)
#H0:Non stationary
#Ha:Stationary
test.unitroot = adf.test(y)
test.unitroot

## Ljung-Box Test (Dependence test):
#H0: The data are independently distributed. 
#Ha: The data are not independently distributed; they exhibit serial correlation.
(test.LB   <- Box.test(y,lag = 10,type = "Ljung-Box"))
(test.LB.abs   <- Box.test(abs(y),lag = 10,type = "Ljung-Box"))
(test.LB.sqr   <- Box.test(y^2,lag = 10,type = "Ljung-Box"))
  
## Tests for the normality (Normality test)
#H0: The data are normal distribution
sh <- shapiro.test(as.numeric(y)) # Shapiro--Wilk
ag <- agostino.test(as.numeric(y)) # D'Agostino's test
jb <- jarque.test(as.numeric(y)) # Jarque--Bera test
cat("p-value: shapiro test",sh$p.value,"    agostino.test", ag$p.value,"    jarque.test", jb$p.value)
stopifnot(min(sh$p.value, ag$p.value, jb$p.value) >= 0.05) 

## Model setting (ARCH(1), GARCH(1,1), TGARCH(1,1))
model.arch<-ugarchspec (variance.model = list(model="sGARCH",garchOrder=c(1,0)), 
                         mean.model = list(armaOrder=c(1,0),include.mean=TRUE), 
                         distribution.model = "std")
model.garch<-ugarchspec(variance.model = list(model="sGARCH",garchOrder=c(1,1)), 
                         mean.model = list(armaOrder=c(1,0),include.mean=TRUE), 
                         distribution.model = "std")
model.egarch<-ugarchspec(variance.model = list(model="eGARCH",garchOrder=c(1,1)), 
                        mean.model = list(armaOrder=c(1,0),include.mean=TRUE), 
                        distribution.model = "std")
model.tgarch<-ugarchspec(variance.model = list(model="fGARCH",garchOrder=c(1,1),submodel="TGARCH"), 
                         mean.model = list(armaOrder=c(1,0),include.mean=TRUE), 
                         distribution.model = "std")

## Modelestimation
# Parameter significance

(fit.arch <- ugarchfit(model.arch, y))
(fit.garch <- ugarchfit(model.garch, y))
(fit.egarch <- ugarchfit(model.egarch, y))
(fit.tgarch <- ugarchfit(model.tgarch, y))


## Likelihood ratio test
# H0: model are good enough
# HA: t innovations necessary
# Decision: We reject the null (in favour of the alternative) if the
#           likelihood-ratio test statistic exceeds the 0.99 quantile of a
#           chi-squared distribution with 1 degree of freedom (1 here as that's
#           the difference in the number of parameters for the two model

(LRT <- 2*(fit.garch@fit$LLH-fit.arch@fit$LLH))
LRT > qchisq(0.99, 1) #=> H0 is rejected
(LRT <- 2*(fit.tgarch@fit$LLH-fit.arch@fit$LLH))
LRT > qchisq(0.99, 1) #=> H0 is rejected
(LRT <- 2*(fit.egarch@fit$LLH-fit.garch@fit$LLH))
LRT > qchisq(0.99, 1) #=> H0 is rejected
(LRT <- 2*(fit.tgarch@fit$LLH-fit.garch@fit$LLH))
LRT > qchisq(0.99, 1) #=> H0 is rejected
(LRT <- 2*(fit.egarch@fit$LLH-fit.arch@fit$LLH))
LRT > qchisq(0.99, 1) #=> H0 is rejected


## Forecast
(forecast1.arch <- ugarchforecast(fit.arch,n.ahead=10))
plot(forecast1.arch,which=3,xlab=" ")
(forecast1.garch <- ugarchforecast(fit.garch,n.ahead=10))
plot(forecast1.garch,which=3)
(forecast1.egarch <- ugarchforecast(fit.egarch,n.ahead=10))
plot(forecast1.egarch,which=3)
(forecast1.tgarch <- ugarchforecast(fit.tgarch,n.ahead=10))
plot(forecast1.tgarch,which=3)

## Residuals
res.arch <- fit.arch@fit[["residuals"]]
res.garch <- fit.garch@fit[["residuals"]]
res.egarch <- fit.garch@fit[["residuals"]]
res.tgarch <- fit.tgarch@fit[["residuals"]]
plot(res.arch)
plot(res.garch)

#ARCH Residuals

param <- coef(fit.arch) # estimated coefficients
sig <- sigma(fit.arch) # estimated volatility
VaR.99 <- quantile(fit.arch, probs = 0.99) # estimated VaR at level 99%
arch <- residuals(fit.arch, standardize = TRUE)

(mu.arch <- mean(arch)) # ok (should be ~= 0)
(sd.arch <- as.vector(sd(arch))) # ok (should be ~= 1)
skewness(arch) # should be 0 (is < 0 => left skewed)
hist(arch) # => left skewed
kurtosis(Z) # should be 6/(nu-4)
nu <- param[["shape"]] # estimated degrees-of-freedom
6/(nu-4) # => sample kurtosis larger than it should be
pt.hat <- function(q) pt((q-mu.Z)/sd.arch, df = nu) # estimated t distribution for Z
ad.test(as.numeric(arch), distr.fun = pt.hat)

#GARCH Residuals

param <- coef(fit.garch) # estimated coefficients
sig <- sigma(fit.garch) # estimated volatility
VaR.99 <- quantile(fit.garch, probs = 0.99) # estimated VaR at level 99%
garch <- residuals(fit.garch, standardize = TRUE)

(mu.garch <- mean(garch)) # ok (should be ~= 0)
(sd.garch <- as.vector(sd(garch))) # ok (should be ~= 1)
skewness(garch) # should be 0 (is < 0 => left skewed)
hist(garch) # => left skewed
kurtosis(garch) # should be 6/(nu-4)
nu <- param[["shape"]] # estimated degrees-of-freedom
6/(nu-4) # => sample kurtosis larger than it should be
pt.hat <- function(q) pt((q-mu.Z)/sd.garch, df = nu) # estimated t distribution for Z
ad.test(as.numeric(garch), distr.fun = pt.hat)

(test.LB   <- Box.test(arch,lag = 10,type = "Ljung-Box"))
(test.LB   <- Box.test(garch,lag = 10,type = "Ljung-Box"))