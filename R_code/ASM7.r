library(quantmod)
library(qrmtools)
library(fGarch)
library(xts)

## Getting data
getSymbols("NFLX")
nflx <- NFLX['2012/2020']
nflx.adj <- nflx$NFLX.Adjusted
y1 <- returns(nflx.adj)
y1=na.omit(y1)
dim(y1)

## Setup
T=length(y1)
p=0.01
WE=30
l1=p*WE
value=1
VaR=matrix(nrow=T,ncol=1)
colnames(VaR)=c('EWMA')


## Setup
lambda=0.94;
s11=var(y1[1:30]);
for (t in (2:WE)) s11= (1-lambda)*y1[t-1]^2+lambda*s11

## Backtest EWMA
for (t in (WE+1):T){
  t1 = t - WE; 
  t2 = t - 1; 
  window = y1[t1:t2] 
  
  s11 = lambda * s11 + (1 - lambda) * y1[t - 1]^2
  VaR[t,1] = -qnorm(p) * sqrt(s11) * value
}

W1=WE+1

# Backtest VR and VaR volatility

VR = sum(y1[W1:T] < -VaR[W1:T])/(p*(T - WE))
s = sd(VaR[W1:T]) # VaR volatility
cat(1,"VR",VR,"VaR vol",s,"\n") # print results

matplot(cbind(y1[W1:T],VaR[W1:T]), type= 'l', xlab = '', ylab = '')
legend("bottomright", bty = "n", lty = c(1),
       col = c("red"),
       legend = c(expression(EWMA)))

# Coverage & Independence test
VaRTest(alpha = 0.05, actual = y1[W1:T], VaR = -VaR[W1:T])
