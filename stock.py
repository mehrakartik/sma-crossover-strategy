# Importing the libraries
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import default_storage
from io import StringIO
import pandas as pd
from pandas_datareader import get_data_yahoo
from datetime import datetime
import plotly.graph_objs as go
from plotly.offline import plot

class StockData:
    def __init__(self, ticker):
        self._ticker = ticker.upper()
        start = datetime(1950, 1, 1)
        end = datetime.today()

        try:
            # Reading data from pandas datareader
            self.stock_df = get_data_yahoo(self.ticker, end = end)
            print(self.ticker)

            # Writing the data into a CSV, in case API read fails
            file = StringIO()
            self.stock_df.to_csv(file)
            file.seek(0)
            try:
                temp = open('Datasets/' + self.ticker + '.csv')
                temp.close()
                from subprocess import run
                run(f"del Datasets\\{self.ticker}.csv", shell = True)
            except:
                pass
            finally:
                default_storage.save('Datasets/' + self.ticker + '.csv', file)
            file.close()
        except:
            self.stock_df = pd.read_csv('Datasets/' + self.ticker + '.csv', index_col = 'Date', parse_dates = True)

        # Start and end date of the stock's data
        self._start_date = self.stock_df.index[0]
        self._end_date = self.stock_df.index[-1]

    # Using property to get start date
    @property
    def start_date(self):
        return self._start_date

    # Using property to get end date
    @property
    def end_date(self):
        return self._end_date

    # Using property to get ticker
    @property
    def ticker(self):
        return self._ticker

    # Return data in date range
    def getDataInRange(self, start = None, end = None):
        # Handling start date
        if not start:
            start = self.start_date
        if start < self.start_date:
            start = self.start_date

        # Handling end date
        if not end:
            end = self.end_date
        elif end > self.end_date:
            end = self.end_date

        return self.stock_df.loc[start:end].copy()

    # Plotting closing prices
    def plotClosingPrice(self, start = None, end = None, fig = None):
        # Ranging DataFrame
        ranged_df = self.getDataInRange(start, end)

        # Data to plot
        data = go.Scatter(x = ranged_df.index, y = ranged_df['Close'], name = self.ticker)
        layout = go.Layout(xaxis = {'title': 'Date'},
                           yaxis = {'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker,
                           xaxis_rangeslider_visible = True)

        return go.Figure(data = data, layout = layout) if not fig else fig.add_trace(data)

    # Simple Moving Average - Crossover Strategy
    def SMA_CS(self, short_window = 40, long_window = 100):
        '''
        Parameters
        ----------
        short_window: Fast moving window
        long_window: Slow moving window
        '''

        # Initialize signals DataFrame with Signal column having values 0
        self.signals = pd.DataFrame(data = 0, index = self.stock_df.index, columns = ['Signal'])

        # Create short and long moving averages columns
        self.signals[f'Short ({short_window} days)'] = self.stock_df['Close'].rolling(window = short_window,
                                                                                  min_periods = 1,
                                                                                  center = False).mean()
        self.signals[f'Long ({long_window} days)'] = self.stock_df['Close'].rolling(window = long_window,
                                                                           min_periods = 1,
                                                                           center = False).mean()

        # Create signals
        from numpy import where
        self.signals['Signal'][short_window:] = where(self.signals[f'Short ({short_window} days)'][short_window:] >
                                                 self.signals[f'Long ({long_window} days)'][short_window:], 1, 0)

        # Trading orders
        self.signals['Positions'] = self.signals['Signal'].diff()

        # Short and long moving averages
        short_avg = go.Scatter(x = self.signals.index, y = self.signals[f'Short ({short_window} days)'],
                                 name = f'Short ({short_window} days)')
        long_avg = go.Scatter(x = self.signals.index, y = self.signals[f'Long ({long_window} days)'],
                              name = f'Long ({long_window} days)')

        # Buy and sell signals
        size = 10
        buy_signal = go.Scatter(x = self.signals[self.signals['Positions'] == 1].index,
                                y = self.signals[self.signals['Positions'] == 1][f'Short ({short_window} days)'],
                                marker = {'symbol': 'triangle-up-dot', 'size': size, 'color': 'green'},
                                mode = 'markers', showlegend = False, hoverinfo = 'skip')
        sell_signal = go.Scatter(x = self.signals[self.signals['Positions'] == -1].index,
                                 y = self.signals[self.signals['Positions'] == -1][f'Short ({short_window} days)'],
                                 marker = {'symbol': 'triangle-down-dot', 'size': size, 'color': 'red'},
                                 mode = 'markers', showlegend = False, hoverinfo = 'skip')

        # Plotting SMAs
        layout = go.Layout(xaxis = {'title': 'Date'},
                           yaxis = {'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker,
                           xaxis_rangeslider_visible = True)
        return go.Figure(data = [short_avg, long_avg, buy_signal, sell_signal], layout = layout)

    # Backtesting
    # Activate this function only after SMA_CS has been executed
    def backtest(self, initial_capital = 1000000, shares = 100):
        # Dataframe 'portfolio' to backtest SMA-CS
        try:
            self.portfolio = pd.DataFrame(index = self.signals.index)
        except AttributeError:
            print('First implement the crossover strategy')
            return

        # Buy 'shares' on the day when short moving average crosses long moving average
        self.portfolio[self.ticker] = shares * self.signals['Signal']

        # Differences in shares owned
        self.portfolio['Diff'] = self.portfolio[self.ticker].diff()

        # Holdings of the shares
        self.portfolio['Holdings'] = self.portfolio[self.ticker].multiply(self.stock_df['Adj Close'], axis = 0)

        # Cash in hand
        self.portfolio['Cash'] = initial_capital - self.portfolio['Diff'].multiply(self.stock_df['Adj Close']).cumsum()

        # Total at an instant
        self.portfolio['Total'] = self.portfolio['Holdings'] + self.portfolio['Cash']

        # Returns of stocks
        self.portfolio['Returns'] = self.portfolio['Total'].pct_change()

        # Plot total
        total = go.Scatter(x = self.portfolio.index, y = self.portfolio['Total'], name = 'Total')

        # Buy and sell signals
        size = 10
        buy_signal = go.Scatter(x = self.portfolio[self.signals['Positions'] == 1].index,
                                y = self.portfolio[self.signals['Positions'] == 1]['Total'],
                                marker = {'symbol': 'triangle-up-dot', 'size': size, 'color': 'green'},
                                mode = 'markers', showlegend = False, hoverinfo = 'skip')
        sell_signal = go.Scatter(x = self.portfolio[self.signals['Positions'] == -1].index,
                                 y = self.portfolio[self.signals['Positions'] == -1]['Total'],
                                 marker = {'symbol': 'triangle-down-dot', 'size': size, 'color': 'red'},
                                 mode = 'markers', showlegend = False, hoverinfo = 'skip')
        layout = go.Layout(xaxis = {'title': 'Date'},
                           yaxis = {'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker,
                           xaxis_rangeslider_visible = True)
        return go.Figure(data = [total, buy_signal, sell_signal], layout = layout)

# Function when accessing hovermode
def onHome(request):
    active_stocks.clear()
    offline.plot(fig, auto_open = False)
    driver = webdriver.PhantomJS(executable_path="phantomjs.exe")
    # driver.set_window_size(1000, 500)
    driver.get('temp-plot.html')
    driver.save_screenshot('my_plot.png')
    return render(request, 'index.html')

# Function when comparing stocks
def onCompare(request):
    # Ticker to be compared to
    ticker = request.GET.get('text', 'default')
    if ticker == 'default':
        return

    # Adding to active stocks
    active_stocks[ticker] = StockData(ticker)

    # Determining best date range
    start = max(active_stocks[ticker].start_date for ticker in active_stocks if ticker != 'fig')
    end = min(active_stocks[ticker].end_date for ticker in active_stocks if ticker != 'fig')

    # Adding ticker trace to current figure
    active_stocks['fig'] = active_stocks[ticker].plotClosingPrice(start = start, end = end,
                                                                  name = ticker, fig = active_stocks['fig'])

    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)

    return render(request, 'Compare.html')

# Function when searching for stocks
def onSubmit(request):
    # Searched ticker
    ticker = request.GET.get('text', 'default')
    if ticker == 'default':
        return

    # Adding to active stocks with figure
    active_stocks[ticker] = StockData(ticker)
    active_stocks['fig'] = active_stocks[ticker].plotClosingPrice()

    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)
    # return render(request, 'Chart.html', {'summary': active_stocks[ticker].summary()})
    return render(request, 'Chart.html')    # Summary function in progress

# Function when clicking on trending stocks
def onTrending(request):
    # Get ticker from image
    ticker = tuple(request.GET.keys())[0].split('.')[0]

    # Adding to active stocks with figure
    active_stocks[ticker] = trending_stocks[ticker]['obj']
    active_stocks['fig'] = trending_stocks[ticker]['fig']
    plot(active_stocks['fig'], filename = 'templates/Page.html', auto_open = False)
    # return render(request, 'Chart.html', {'summary': active_stocks[ticker].summary()})
    return render(request, 'Page.html')    # Summary function in progress

# Function for Simple Moving Average Crossover Strategy
def onSMA(request):
    # Slow (long) and fast (short) moving windows
    short_window = request.GET.get('short_window', 'default')
    long_window = request.GET.get('long_window', 'default')
    if short_window == 'default' or long_window == 'default':
        return

    for key in active_stocks:
        if key != 'fig':
            plot(active_stocks[key].SMA_CS(short_window = short_window, long_window = long_window),
            filename = 'Page.html', auto_open = False)

    return render(request, 'SMA.html')

# Function when removing SMA option
def onRemoveSMA(request):
    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)
    return render(request, 'Chart.html')

# Function when removing comparison stock
def onRemoveComparison(request):
    # Ticker to remove
    ticker = None # Pass ticker of button to remove

    # Removing from active stocks and figure as well
    active_stocks.pop(ticker)
    active_stocks['fig'].data = tuple(filter(lambda stock: stock.name != ticker, active_stocks['fig'].data))

    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)

    return render(request, 'Compare.html') if len(active_stocks['fig'].data) > 1 else render(request, 'Chart.html')

# Currently active stocks and figure
active_stocks = {}
trending_stocks = {'GOOG': {'obj': StockData('GOOG')},
                    'AAPL': {'obj': StockData('AAPL')},
                    'AMZN': {'obj': StockData('AMZN')},
                    'MSFT': {'obj': StockData('MSFT')}}
for key in trending_stocks:
    trending_stocks[key]['fig'] = trending_stocks[key]['obj'].plotClosingPrice()
