library("quantmod")
library("qrmtools")
library("rugarch")
library("fGarch")

## Data
getSymbols("AAPL")
getSymbols("AMZN")
getSymbols("FB")
aapl<-na.omit(as.numeric(returns(AAPL$AAPL.Close['2012/2020'])))
amzn<-na.omit(as.numeric(returns(AMZN$AMZN.Close['2012/2020'])))
fb<-na.omit(as.numeric(returns(FB$FB.Close['2012/2020'])))
y<-aapl
y1<-fb
y2 <-amzn

## Backtest for Apple
T = length(y) 
WE = 500 # window length
p = 0.01 # probability
value = 1; # portfolio
l1 = p*WE
VaR = matrix(nrow=T,ncol=3) 

## Calculate VR and VaR Volatility
for (t in (WE + 1):T){
  t1 = t - WE; 
  t2 = t - 1; 
  window = y[t1:t2] # data for estimation
  # HS
  ys = sort(window) # sort returns
  VaR[t,1] = -ys[l1]* value # VaR number
  # GARCH(1,1)-norm
  g=garchFit(formula = ~ garch(1,1), window, cond.dist = c("norm"), trace=FALSE, include.mean=FALSE)
  par=g@fit$matcoef # put parameters into vector par
  s4=par[1]+par[2]* window[WE]^2+par[3]* g@h.t[WE]
  VaR[t,2] = -sqrt(s4) * qnorm(p) * value
  # GARCH(1,1)-std
  g=garchFit(formula = ~ garch(1,1), window, cond.dist = c("std"), trace=FALSE, include.mean=FALSE)
  par=g@fit$matcoef # put parameters into vector par
  s4=par[1]+par[2]* window[WE]^2+par[3]* g@h.t[WE]
  VaR[t,3] = -sqrt(s4) * qnorm(p) * value
}

W1 = WE+1
for (i in 1:3){
  VR = sum(googl[W1:T] < -VaR[W1:T,i])/(p*(T - WE))
  s = sd(VaR[W1:T,i]) # VaR volatility
  cat(i,"VR",VR,"VaR vol",s,"\n") # print results
}
# Graph
matplot(cbind(googl[W1:T],VaR[W1:T,1],VaR[W1:T,2],VaR[W1:T,3]),type="l",ylab = "", main="Backtest 1% VaR for AAPL")
legend("topleft", bty = "n", lty = c(1,1),
       col = c("red", "green","blue"),
       legend = c(expression(HS), expression(GARCH_norm), expression(GARCH_std)))

## Backtest for Facebook
T = length(y1) 
WE = 500 # window length
p = 0.01 # probability
value = 1; # portfolio
l1 = p*WE
VaR = matrix(nrow=T,ncol=3) 

## Calculate VaR and VaR volatility
for (t in (WE + 1):T){
  t1 = t - WE; 
  t2 = t - 1; 
  window = y1[t1:t2] # data for estimation
  # HS
  ys = sort(window) # sort returns
  VaR[t,1] = -ys[l1]* value # VaR number
  # GARCH(1,1)-norm
  g=garchFit(formula = ~ garch(1,1), window, cond.dist = c("norm"), trace=FALSE, include.mean=FALSE)
  par=g@fit$matcoef # put parameters into vector par
  s4=par[1]+par[2]* window[WE]^2+par[3]* g@h.t[WE]
  VaR[t,2] = -sqrt(s4) * qnorm(p) * value
  # GARCH(1,1)-std
  g=garchFit(formula = ~ garch(1,1), window, cond.dist = c("std"), trace=FALSE, include.mean=FALSE)
  par=g@fit$matcoef # put parameters into vector par
  s4=par[1]+par[2]* window[WE]^2+par[3]* g@h.t[WE]
  VaR[t,3] = -sqrt(s4) * qnorm(p) * value
}

W1 = WE+1
for (i in 1:3){
  VR = sum(googl[W1:T] < -VaR[W1:T,i])/(p*(T - WE))
  s = sd(VaR[W1:T,i]) # VaR volatility
  cat(i,"VR",VR,"VaR vol",s,"\n") # print results
}
# Graph
matplot(cbind(googl[W1:T],VaR[W1:T,1],VaR[W1:T,2],VaR[W1:T,3]),type="l",ylab = "", main="Backtest 1% VaR for FB")
legend("topleft", bty = "n", lty = c(1,1),
       col = c("red", "green","blue"),
       legend = c(expression(HS), expression(GARCH_norm), expression(GARCH_std)))

## Backtest for Amazon
T = length(y2) 
WE = 500 # window length
p = 0.01 # probability
value = 1; # portfolio
l1 = p*WE
VaR = matrix(nrow=T,ncol=3) 

## Calculate VaR and VaR volatility
for (t in (WE + 1):T){
  t1 = t - WE; 
  t2 = t - 1; 
  window = y2[t1:t2] # data for estimation
  # HS
  ys = sort(window) # sort returns
  VaR[t,1] = -ys[l1]* value # VaR number
  # GARCH(1,1)-norm
  g=garchFit(formula = ~ garch(1,1), window, cond.dist = c("norm"), trace=FALSE, include.mean=FALSE)
  par=g@fit$matcoef # put parameters into vector par
  s4=par[1]+par[2]* window[WE]^2+par[3]* g@h.t[WE]
  VaR[t,2] = -sqrt(s4) * qnorm(p) * value
  # GARCH(1,1)-std
  g=garchFit(formula = ~ garch(1,1), window, cond.dist = c("std"), trace=FALSE, include.mean=FALSE)
  par=g@fit$matcoef # put parameters into vector par
  s4=par[1]+par[2]* window[WE]^2+par[3]* g@h.t[WE]
  VaR[t,3] = -sqrt(s4) * qnorm(p) * value
}

W1 = WE+1
for (i in 1:3){
  VR = sum(googl[W1:T] < -VaR[W1:T,i])/(p*(T - WE))
  s = sd(VaR[W1:T,i]) # VaR volatility
  cat(i,"VR",VR,"VaR vol",s,"\n") # print results
}
# Graph
matplot(cbind(googl[W1:T],VaR[W1:T,1],VaR[W1:T,2],VaR[W1:T,3]),type="l",ylab = "", main="Backtest 1% VaR for AMZN")
legend("topleft", bty = "n", lty = c(1,1),
       col = c("red", "green","blue"),
       legend = c(expression(HS), expression(GARCH_norm), expression(GARCH_std)))

