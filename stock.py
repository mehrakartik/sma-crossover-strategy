# Importing the libraries
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot

class StockData:
    def __init__(self, ticker):
        self._ticker = ticker.upper()
        start = datetime(1900, 1, 1)
        end = datetime.today()

        try:
            # Reading data from pandas datareader
            self.stock_df = pdr.get_data_yahoo(self.ticker, start = start, end = end)

            # Writing the data into a CSV, in case API read fails
            self.stock_df.to_csv('Datasets/' + self.ticker + '.csv')
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
        data = go.Scatter(x = ranged_df.index, y = ranged_df['Close'], name = name)
        layout = go.Layout(xaxis = {'title': 'Date'},
                           yaxis = {'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker)
        fig.update_layout()

        return go.Figure(data = data, layout = layout) if not fig else fig.add_trace(data)

    # Moving Windows
    def movingWindows(self, short_window = 40, long_window  = 100):
        '''
        Parameters
        ----------
        short_window: Fast moving window
        long_window: Slow moving window
        '''

        # Temporary DataFrame
        temp_df = self.stock_df.copy()

        # Short and long moving windows rolling mean
        temp_df[f'{short_window} days'] = temp_df['Adj Close'].rolling(window = short_window).mean()
        temp_df[f'{long_window} days'] = temp_df['Adj Close'].rolling(window = long_window).mean()

        # Plot adjusted close price, short and long windows rolling means
        layout = go.Layout(xaxis = {'showgrid': kwargs['grid'], 'title': 'Date'},
                           yaxis = {'showgrid': kwargs['grid'], 'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker)
        adj_trace = go.Scatter(x = temp_df.index, y = temp_df['Adj Close'], name = 'Adj Close')
        short_trace = go.Scatter(x = temp_df.index, y = temp_df[f'{short_window} days'], name = f'{short_window} days')
        long_trace = go.Scatter(x = temp_df.index, y = temp_df[f'{long_window} days'], name = f'{long_window} days')
        return go.Figure(data = [adj_trace, short_trace, long_trace], layout = layout)

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
                           hovermode = 'x', title = self.ticker)
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
                           hovermode = 'x', title = self.ticker)
        return go.Figure(data = [total, buy_signal, sell_signal], layout = layout)

def onCompare(request):
    ticker = request.GET.get('text', 'default')
    if ticker == 'default':
        return
    active_stocks[ticker] = StockData(ticker)
    start_dates = []
    end_dates = []
    for key in active_stocks:
        if key == 'fig':
            continue
        start_dates.append(active_stocks[key].start_date)
        end_dates.append(active_stocks[key].end_date)
        start = max(start_dates)
        end = min(end_dates)
    active_stocks['fig'] = active_stocks[ticker].plotClosingPrice(start = start, end = end,
                                                                  name = ticker, fig = active_stocks['fig'])
    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)
    return render(request, 'Compare.html')

def onSubmit(request):
    ticker = request.GET.get('text', 'default')
    if ticker == 'default':
        return
    active_stocks[ticker] = StockData(ticker)
    active_stocks['fig'] = active_stocks[ticker].plotClosingPrice(name = ticker)
    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)
    # return render(request, 'Chart.html', {'summary': active_stocks[ticker].summary()})
    return render(request, 'Chart.html')    # Summary function in progress

def onMovingWindows(request):
    short_window = request.GET.get('short_window', 'default')
    long_window = request.GET.get('long_window', 'default')
    if short_window == 'default' or long_window == 'default':
        return
    for key in active_stocks:
        if key != 'fig':
            plot(active_stocks[key].movingWindows(), filename = 'Page.html', auto_open = False)
    return render(request, 'MovingWindows.html')

def onSMA(request):
    short_window = request.GET.get('short_window', 'default')
    long_window = request.GET.get('long_window', 'default')
    if short_window == 'default' or long_window == 'default':
        return
    for key in active_stocks:
        if key != 'fig':
            plot(active_stocks[key].SMA_CS(), filename = 'Page.html', auto_open = False)
    return render(request, 'SMA.html')

def onRemoveMovingOrSMA(request):
    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)
    return render(request, 'Chart.html')

def onRemoveComparison(request):
    ticker = None # Pass ticker of button to remove
    active_stocks.pop(ticker)
    active_stocks['fig'].data = tuple(filter(lambda stock: stock.name != ticker, active_stocks['fig'].data))
    plot(active_stocks['fig'], filename = 'Page.html', auto_open = False)
    return render(request, 'Compare.html') if len(active_stocks['fig'].data) > 1 else render(request, 'Chart.html')

# Currently active stocks and figure
active_stocks = {}

if __name__ == 'main':
    pass
