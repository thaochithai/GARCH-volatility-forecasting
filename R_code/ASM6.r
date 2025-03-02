library(quantmod)
library(qrmtools)
library(fGarch)
library(xts)
library(zoo)
library(rugarch)

## Data
getSymbols("NFLX")
getSymbols("ROKU")
getSymbols("DIS")

## Portfolio
port <-cbind(NFLX$NFLX.Adjusted,ROKU$ROKU.Adjusted,DIS$DIS.Adjusted)
port <-port["2012/2020"]
y <- returns(port)
dim(y)
w<-matrix(c(1/4,1/4,1/2))
yp<- y %*% w
yp<-na.omit(yp)
summary(yp)

## Setup
T = length(yp)
WE = 30
p = 0.01
l1 = WE * p 
value = 1; 
VaR = matrix(nrow=T,ncol=4)

## Setup
lambda = 0.94;
s11 = var(yp[1:30]);
for(t in 2:WE) s11 = lambda * s11 + (1 - lambda) * yp[t - 1]^2

for (t in (WE + 1):T){
  t1 = t - WE; # start of the data window
  t2 = t - 1; # end of the data window
  window = yp[t1:t2] # data for estimation
# EWMA backtest
  s11 = lambda * s11 + (1 - lambda) * yp[t - 1]^2
  VaR[t,1] = -qnorm(p) * sqrt(s11) * value
}

W1=WE+1

# Backtest VR and VaR volatility
  VR = sum(yp[W1:T] < -VaR[W1:T])/(p*(T - WE))
  s = sd(VaR[W1:T]) # VaR volatility
  cat(1,"VR",VR,"VaR vol",s,"\n") # print results

  matplot(cbind(yp[W1:T],VaR[W1:T]), type= 'l', xlab = '', ylab = '')
legend("bottomright", bty = "n", lty = c(1),
       col = c("red"),
       legend = c(expression(EWMA)))
# Coverage test
VaRTest(alpha = 0.05, actual = yp[W1:T], VaR = -VaR[W1:T])
