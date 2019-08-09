# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
import quandl
from datetime import datetime

ticker = 'GOOG'
try:
    # Reading data from pandas datareader
    df = pdr.get_data_yahoo(ticker, start = datetime(2004, 8, 19), end = datetime.today())

    # Writing the data into a CSV, in case API read fails
    df.to_csv('Datasets/' + ticker + '.csv')
except:
    df = pd.read_csv('Datasets/' + ticker + '.csv', index_col = 'Date', parse_dates = True)

# Earliest 5 rows
df.tail()

# Latest 5 rows
df.head()

df.describe()

# Sampling the data month-wise
mw = df.resample('M').mean().reset_index()
mw['Date'] = mw['Date'].apply(lambda date: f'{date.year}-{date.month}')
mw.set_index('Date', inplace = True)
mw.head()

# Plotting closing prices
df['Close'].plot(grid = True, figsize = (12, 7))
plt.plot()
plt.show()

# Daily percentage change
daily_pct_change = df['Adj Close'].pct_change().fillna(0)
# Another method
# daily_pct_change = (df['Adj Close'] - df['Adj Close'].shift()) / df['Adj Close'].shift()

# Daily log returns
daily_log_returns = np.log(daily_pct_change)

# Monthly percentage change
monthly = df.resample(rule = 'M').mean().pct_change()

# Quarter percentage change
quarterly = df.resample(rule = '4M').mean().pct_change()

# Daily percentage change is normally distributed and most of the changes in the proximity of 0
daily_pct_change.hist(bins = 50, figsize = (8, 5))
plt.plot()

# # Cumulative daily rate of return
# The cumulative daily rate of return is useful to determine the value of an investment at regular intervals.
# You can calculate the cumulative daily rate of return by using the daily percentage change values, adding 1 to
# them and calculating the cumulative product with the resulting values

cum_daily_return = (1 + daily_pct_change).cumprod()
cum_daily_return.plot(figsize = (12, 7))
plt.show()

# Cumulative monthly returns
cum_monthly_returns = cum_daily_return.resample(rule = 'M').mean()
cum_monthly_returns.plot(figsize = (12, 7))
plt.show()

# Function to get multiple stock data
def get(tickers, start, end):
    def data(ticker):
        return pdr.get_data_yahoo(ticker, start = start, end = end)
    return pd.concat(map(data, tickers), keys = tickers, names = ['Ticker', 'Date'])

data_list = get(tickers = ['AAPL', 'GOOG', 'MSFT', 'IBM'], start = datetime(2010, 1, 1), end = datetime.today())
data_list.head()

daily_adj_close = data_list['Adj Close'].reset_index().pivot(index = 'Date',
                                                             columns = 'Ticker',
                                                             values = 'Adj Close')
daily_adj_close.head()

# Daily percentage change for each stock
daily_pct_change = daily_adj_close.pct_change()
daily_pct_change.hist(figsize = (15, 10), bins = 50, sharex = True, sharey = True)
plt.show()

# Scatter plot of daily adj close for all stocks
sns.pairplot(daily_adj_close, diag_kind = 'kde')
# Another way
# pd.plotting.scatter_matrix(daily_adj_close, figsize = (15, 10), alpha = 0.1, diagonal = 'kde')
plt.show()

# # Moving Windows
# Moving windows are there when you compute the statistic on a window of data represented by a particular
# period of time and then slide the window across the data by a specified interval. That way, the statistic
# is continually calculated as long as the window falls first within the dates of the time series.

short_window = 40
long_window = 200

# Short and long moving windows rolling mean
df[f'{short_window}'] = df['Adj Close'].rolling(window = short_window).mean()
df[f'{long_window}'] = df['Adj Close'].rolling(window = long_window).mean()

# Plot adjusted close price, short and long windows rolling means
df[['Adj Close', f'{short_window}', f'{long_window}']].plot(figsize = (15, 10))
plt.show()

# # Volatility of a stock
# The volatility of a stock is a measurement of the change in variance in the returns of a stock over a
# specific period of time. It is common to compare the volatility of a stock with another stock to get a
# feel for which may have less risk or to a market index to examine the stock’s volatility in the overall
# market. Generally, the higher the volatility, the riskier the investment in that stock, which results in
# investing in one over another.

min_periods = 75
x = np.random.randint(1, daily_pct_change.shape[0] - 200)
vol = daily_pct_change[x:x+200].rolling(window = min_periods).mean().dropna(how = 'all') * np.sqrt(min_periods)
vol.plot(figsize = (15, 10))
plt.show()

# # Ordinary Least-Squares Regression (OLS)
# In statistics, ordinary least squares (OLS) is a type of linear least squares method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear function of a set of explanatory variables by the principle of least squares: minimizing the sum of the squares of the differences between the observed dependent variable (values of the variable being predicted) in the given dataset and those predicted by the linear function.

# # Note: Values not updated with current results.

# - The number of observations (No. Observations). Note that you could also derive this with the Pandas package by using the info() function. Run return_data.info() in the IPython console of the DataCamp Light chunk above to confirm this.
# - The degree of freedom of the residuals (DF Residuals)
# - The number of parameters in the model, indicated by DF Model; Note that the number doesn’t include the constant term X which was defined in the code above.

# ## This was basically the whole left column that you went over. The right column gives you some more insight into the goodness of the fit. You see, for example:

# - R-squared, which is the coefficient of determination. This score indicates how well the regression line approximates the real data points. In this case, the result is 0.280. In percentages, this means that the score is at 28%. When the score is 0%, it indicates that the model explains none of the variability of the response data around its mean. Of course, a score of 100% indicates the opposite.
# - You also see the Adj. R-squared score, which at first sight gives the same number. However, the calculation behind this metric adjusts the R-Squared value based on the number of observations and the degrees-of-freedom of the residuals (registered in DF Residuals). The adjustment in this case hasn’t had much effect, as the result of the adjusted score is still the same as the regular R-squared score.
# - The F-statistic measures how significant the fit is. It is calculated by dividing the mean squared error of the model by the mean squared error of the residuals. The F-statistic for this model is 514.2.
# - Next, there’s also the Prob (F-statistic), which indicates the probability that you would get the result of the F-statistic, given the null hypothesis that they are unrelated.
# - The Log-likelihood indicates the log of the likelihood function, which is, in this case 3513.2.
# - The AIC is the Akaike Information Criterion: this metric adjusts the log-likelihood based on the number of observations and the complexity of the model. The AIC of this model is -7022.
# - Lastly, the BIC or the Bayesian Information Criterion, is similar to the AIC that you just have seen, but it penalizes models with more parameters more severely. Given the fact that this model only has one parameter (check DF Model), the BIC score will be the same as the AIC score.

# ## Below the first part of the model summary, you see reports for each of the model’s coefficients:

# - The estimated value of the coefficient is registered at coef.
# - std err is the standard error of the estimate of the coefficient.
# - There’s also the t-statistic value, which you’ll find under t. This metric is used to measure how statistically significant a coefficient is.
# - P > |t| indicates the null-hypothesis that the coefficient = 0 is true. If it is less than the confidence level, often 0.05, it indicates that there is a statistically significant relationship between the term and the response. In this case, you see that the constant has a value of 0.198, while AAPL is set at 0.000.

# - Lastly, there is a final part of the model summary in which you’ll see other statistical tests to assess the distribution of the residuals:

# - Omnibus, which is the Omnibus D’Angostino’s test: it provides a combined statistical test for the presence of skewness and kurtosis.
# - The Prob(Omnibus) is the Omnibus metric turned into a probability.
# - Next, the Skew or Skewness measures the symmetry of the data about the mean.
# - The Kurtosis gives an indication of the shape of the distribution, as it compares the amount of data close to the mean with those far away from the mean (in the tails).
# - Durbin-Watson is a test for the presence of autocorrelation, and the Jarque-Bera is another test of the skewness and kurtosis. You can also turn the result of this test into a probability, as you can see in Prob (JB).
# - Lastly, you have the Cond. No, which tests the multicollinearity.

# Linear Regression implementation using statsmodels

import statsmodels.api as sm

# Isolate adjusted closing prices
all_adj_close = data_list[['Adj Close']]

# Log returns
all_log_returns = np.log(all_adj_close / all_adj_close.shift())

aapl = all_log_returns[all_log_returns.index.get_level_values('Ticker') == 'AAPL'].droplevel('Ticker')
msft = all_log_returns[all_log_returns.index.get_level_values('Ticker') == 'MSFT'].droplevel('Ticker')
returns = pd.concat([aapl, msft], axis = 1)[1:]
returns.columns = ['AAPL', 'MSFT']
X = sm.add_constant(returns['AAPL'])
model = sm.OLS(returns['MSFT'], X).fit()
model.summary()

# Visualizing OLS
plt.grid(True)
plt.scatter(returns['AAPL'], returns['MSFT'], s = 2.5 ** 2)
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft Returns')
ax = plt.axis()
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, model.params[0] + model.params[1] * x, color = 'red')
plt.show()

# Linear Regression implementation using sklearn.linear_model

X = returns['AAPL'].values.reshape(-1, 1)
y = returns['MSFT'].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

plt.grid(True)
plt.scatter(X, y, marker = '.', s = 5 ** 2)
ax = plt.axis()
x = np.linspace(ax[0], ax[1] + 0.01)
# regressor.intercept_: b0
# regressor.coef_: b1
plt.plot(x, regressor.intercept_ + regressor.coef_ * x, color = 'red')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft Returns')
plt.show()

# Initialize short and long windows
short_window = 40
long_window = 100

# Initialize signals DataFrame with Signal column having values 0
signals = pd.DataFrame(data = 0, index = df.index, columns = ['Signal'])

# Create short and long moving averages columns
signals['Short Moving Average'] = df['Close'].rolling(window = short_window,
                                                      min_periods = 1, center = False).mean()
signals['Long Moving Average'] = df['Close'].rolling(window = long_window, min_periods = 1, center = False).mean()

# Create signals
signals['Signal'][short_window:] = np.where(signals['Short Moving Average'][short_window:] >
                             signals['Long Moving Average'][short_window:], 1, 0)

# Trading orders
signals['Positions'] = signals['Signal'].diff()

signals.head()

# Plotting short and long moving averages
signals[['Short Moving Average', 'Long Moving Average']].plot(figsize = (15, 10), color = ['skyblue', 'orange'])
plt.ylabel('Price in $')

# Plotting buy signals
plt.scatter(signals[signals['Positions'] == 1].index,
         signals[signals['Positions'] == 1]['Short Moving Average'],
         marker = '^', s = 9 ** 2, color = 'green')

# Plotting sell signals
plt.scatter(signals[signals['Positions'] == -1].index,
         signals[signals['Positions'] == -1]['Short Moving Average'],
         marker = 'v', s = 9 ** 2, color = 'red')
plt.show()

# Backtesting

# Notes:
# 1. initial_capital: Capital (asset) at the time of investment.
# 2. shares: Number of shares to buy or sell at every crossover.
# 3. portfolio: A Pandas DataFrame to backtest SMA-CS.
# 4. portfolio[ticker]: Number of shares to buy when the signal is 1 (short moving average > long moving average).
# 5. portfolio['Diff']: Number of shares owned compared with previous day.
# 6. portfolio['Holdings']: Amount of money in shares.
# 7. portfolio['Cash']: Cash available with the user. It is calculated as:
#         Initial capital - Cumulative_Sum(Number of shares owned at an instant in a day * Adjusted Closing Price at that day)
#         Cumulative Sum is taken to ensure that the previous transactions are taken into account. This helps in calculating total at any instant.
# 8. portfolio['Total']: Holdings in shares (Liabilities) + Cash (Assets)
# 9. portfolio['Returns']: Returns of the investment per day.

# Set the initial capital
initial_capital = 1000000

# Shares to buy on the day when short moving average crosses long moving average
shares = 100

# Dataframe 'portfolio' to backtest SMA-CS
portfolio = pd.DataFrame(index = signals.index)

# Buy 'shares' on the day when short moving average crosses long moving average
portfolio[ticker] = shares * signals['Signal']

# Differences in shares owned
portfolio['Diff'] = portfolio[ticker].diff()

# Holdings of the shares
portfolio['Holdings'] = portfolio[ticker].multiply(df['Adj Close'], axis = 0)

# Cash in hand
portfolio['Cash'] = initial_capital - portfolio['Diff'].multiply(df['Adj Close']).cumsum()

# Total at an instant
portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']

# Returns of stocks
portfolio['Returns'] = portfolio['Total'].pct_change()

portfolio.iloc[0:60, :]

portfolio['Total'].plot(figsize = (15, 10))
plt.ylabel('Portfolio Value in $')

# Plotting buy signals
plt.scatter(portfolio[signals['Positions'] == 1].index,
            portfolio[signals['Positions'] == 1]['Total'],
            marker = '^', s = 9 ** 2, color = 'green')

# Plotting sell signals
plt.scatter(portfolio[signals['Positions'] == -1].index,
            portfolio[signals['Positions'] == -1]['Total'],
            marker = 'v', s = 9 ** 2, color = 'red')
plt.show()