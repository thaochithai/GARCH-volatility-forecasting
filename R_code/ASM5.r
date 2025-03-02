library("zoo")
library("xts")
library("quantmod")
library("qrmtools")
library("rugarch")

## Data
getSymbols("DELL")
getSymbols("IBM")
getSymbols("INTC")
dell<-na.omit(as.numeric(returns(DELL$DELL.Adjusted['2012/2020'])))
ibm<-na.omit(as.numeric(returns(IBM$IBM.Adjusted['2012/2020'])))
intel<-na.omit(as.numeric(returns(INTC$INTC.Adjusted['2012/2020'])))
y.dell<-dell
y.ibm <- ibm
y.intel <- intel
plot.zoo(y.dell)
plot.zoo(y.ibm)
plot.zoo(y.intel)

## Backtest for Dell

#VaR backtest
GARCH.t <- ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
                      variance.model = list(model = "sGARCH",  garchOrder = c(1,1)),
                      distribution.model = "std")
roll <- ugarchroll(GARCH.t, y.dell, n.start = 500, window.size = 500, refit.every = 1,
                     refit.window = "moving", solver = "hybrid", calculate.VaR = TRUE, VaR.alpha = 0.01)
report(roll, type = "VaR",VaR.alpha = 0.01, conf.level = 0.99)

VaR.table <- as.data.frame(roll, which = "VaR")
names(VaR.table)
realized <- VaR.table[,"realized"]
VaR01 = VaR.table[,1]

VaRTest(alpha = 0.01, actual = realized, VaR = VaR01)
par.n <- as.data.frame(roll, which = "density")
names(par.n)

#ES backtest
ES01 = par.n$Mu - par.n$Sigma*ES_t(0.99, df = Inf)
ESTest(alpha = 0.01, actual = realized, ES = ES01, VaR = VaR01)

## Backtest for IBM

#VaR backtest
GARCH.t <- ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
                      variance.model = list(model = "sGARCH",  garchOrder = c(1,1)),
                      distribution.model = "std")
roll <- ugarchroll(GARCH.t, y.ibm, n.start = 500, window.size = 500, refit.every = 1,
                   refit.window = "moving", solver = "hybrid", calculate.VaR = TRUE, VaR.alpha = 0.01)
report(roll, type = "VaR",VaR.alpha = 0.01, conf.level = 0.99)

VaR.table <- as.data.frame(roll, which = "VaR")
names(VaR.table)
realized <- VaR.table[,"realized"]
VaR01 = VaR.table[,1]

VaRTest(alpha = 0.01, actual = realized, VaR = VaR01)
par.n <- as.data.frame(roll, which = "density")
names(par.n)

#ES backtest
ES01 = par.n$Mu - par.n$Sigma*ES_t(0.99, df = Inf)
ESTest(alpha = 0.01, actual = realized, ES = ES01, VaR = VaR01)

## Backtest for Intel

#VaR backtest
GARCH.t <- ugarchspec(mean.model = list(armaOrder = c(0,0), include.mean = FALSE),
                      variance.model = list(model = "sGARCH",  garchOrder = c(1,1)),
                      distribution.model = "std")
roll <- ugarchroll(GARCH.t, y.intel, n.start = 500, window.size = 500, refit.every = 1,
                   refit.window = "moving", solver = "hybrid", calculate.VaR = TRUE, VaR.alpha = 0.01)
report(roll, type = "VaR",VaR.alpha = 0.01, conf.level = 0.99)

VaR.table <- as.data.frame(roll, which = "VaR")
names(VaR.table)
realized <- VaR.table[,"realized"]
VaR01 = VaR.table[,1]

VaRTest(alpha = 0.01, actual = realized, VaR = VaR01)
par.n <- as.data.frame(roll, which = "density")
names(par.n)

#ES backtest
ES01 = par.n$Mu - par.n$Sigma*ES_t(0.95, df = Inf)
ESTest(alpha = 0.01, actual = realized, ES = ES01, VaR = VaR01)


