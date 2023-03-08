import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


class portfolio:

    # Download the historical data of stock A, B and C and calculate the annualized return
    # of the stocks, and get the risk-free rate
    def __init__(self, stockA, stockB, stockC, start, end, Rf):
        self.stockA = yf.download(stockA, start, end)
        self.stockA["Return"] = np.log(self.stockA["Close"] / self.stockA["Close"].shift())
        self.stockA.dropna(inplace=True)
        self.stockA_annualized_return = np.expm1(252 * self.stockA["Return"].mean())
        self.stockB = yf.download(stockB, start, end)
        self.stockB["Return"] = np.log(self.stockB["Close"] / self.stockB["Close"].shift())
        self.stockB.dropna(inplace=True)
        self.stockB_annualized_return = np.expm1(252 * self.stockB["Return"].mean())
        self.stockC = yf.download(stockC, start, end)
        self.stockC["Return"] = np.log(self.stockC["Close"] / self.stockC["Close"].shift())
        self.stockC.dropna(inplace=True)
        self.stockC_annualized_return = np.expm1(252 * self.stockC["Return"].mean())
        self.Rf = Rf
        self.Return = [self.stockA_annualized_return, self.stockB_annualized_return, self.stockC_annualized_return]
        self.Returns = [self.stockA["Return"], self.stockB["Return"], self.stockC["Return"]]
        self.portfolio_return = []
        self.portfolio_SD = []
        self.portfolio_Sharpe = []

# Try different weightings of stock A, B and C to get the efficient frontier, then construct a CML
    def get_efficient_frontier_and_CML(self):
        for i in np.arange(-2, 2.001, 0.01):
            for j in np.arange(-2, 2.001, 0.01):
                for k in np.arange(-2, 2.001, 0.01):
                    if round(i + j + k, 3) == 1:
                        weight = [i, j, k]
                        self.portfolio_return.append(np.matmul(weight, self.Return))
                        self.portfolio_SD.append(
                            np.sqrt(252) * np.expm1(np.sqrt(np.matmul(np.matmul(weight, np.cov(self.Returns)),
                                                                  np.transpose(weight)))))
                        self.portfolio_Sharpe.append((np.matmul(weight, self.Return)-self.Rf)/(
                            np.sqrt(252) * np.expm1(np.sqrt(np.matmul(np.matmul(weight, np.cov(self.Returns)),
                                                                  np.transpose(weight))))))
        max_sharpe = np.max(self.portfolio_Sharpe)
        var = np.linspace(0, np.max(self.portfolio_SD), 1000)
        CML = self.Rf + var * max_sharpe
        plt.plot(var, CML, linewidth=1, c="orange")
        plt.scatter(self.portfolio_SD, self.portfolio_return, s=0.05)
        plt.xlim([0, np.max(self.portfolio_SD)+0.1])
        plt.show()

# example
Portfolio = portfolio("TSLA", "MCD", "XOM","2019-01-01", "2022-01-01", 0.04)
Portfolio.get_efficient_frontier_and_CML()
