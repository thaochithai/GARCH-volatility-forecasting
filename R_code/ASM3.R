library("zoo")
library("xts")
library("quantmod")
library("qrmtools")
library("fGarch")

## create portfolio
getSymbols("FB")
getSymbols("AAPL")
getSymbols("TSLA")

p <-cbind(FB$FB.Adjusted,AAPL$AAPL.Adjusted,TSLA$TSLA.Adjusted)
p <-p["2012-01-01/2020"]
y <- returns(p)
dim(y)
w<-matrix(c(1/3,1/3,1/3))
yp1<- y %*% w
yp1<-na.omit(yp1)
summary(yp1)
sd(yp1)
skewness(yp1)
kurtosis(yp1)

## Setup
T = length(yp1) # number of observations for return y
WE = 1000 # estimation window length
p = 0.01 # probability
l1 = WE * p # HS observation
value = 1; # portfolio
VaR = matrix(nrow=T,ncol=4) # matrix to hold VaR forecasts for 4 models
# EWMA setup
lambda = 0.94;
s11 = var(yp1[1:30]);
for(t in 2:WE) s11 = lambda * s11 + (1 - lambda) * yp1[t - 1]^2

for (t in (WE + 1):T){
  t1 = t - WE; # start of the data window
  t2 = t - 1; # end of the data window
  window = yp1[t1:t2] # data for estimation
  # EWMA
  s11 = lambda * s11 + (1 - lambda) * yp1[t - 1]^2
  VaR[t,1] = -qnorm(p) * sqrt(s11) * value
  # MA
  VaR[t,2] = -sd(window) * qnorm(p)* value
  # HS
  ys = sort(window) # sort returns
  VaR[t,3] = -ys[l1]* value # VaR number
  # GARCH(1,1)
  g=garchFit(formula = ~ garch(1,1), window ,trace=FALSE,
             include.mean=FALSE)
  par=g@fit$matcoef # put parameters into vector par
  s4=par[1]+par[2]* window[WE]^2+par[3]* g@h.t[WE]
  VaR[t,4] = -sqrt(s4) * qnorm(p) * value
}

W1 = WE+1
for (i in 1:4){
  VR = sum(yp1[W1:T] < -VaR[W1:T,i])/(p*(T - WE))
  s = sd(VaR[W1:T,i]) # VaR volatility
  cat(i,"VR",VR,"VaR vol",s,"\n") # print results
}
matplot(cbind(yp1[W1:T],VaR[W1:T,1],VaR[W1:T,3]),type="l",ylab = "")
legend("bottomright", bty = "n", lty = c(1,1),
       col = c("red", "green"),
       legend = c(expression(EWMA), expression(HS)))

