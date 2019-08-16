# Importing the libraries
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.views.decorators.cache import never_cache
from requests.exceptions import ConnectionError
from io import StringIO
from os.path import exists, join
from os import remove, getcwd
from gc import collect
import pandas as pd
from pandas_datareader import get_data_yahoo
from pandas_datareader._utils import RemoteDataError
from datetime import datetime
import plotly.graph_objs as go
from plotly.offline import plot

class StockData:
    def __init__(self, ticker):
        self.remote_data_error = False
        self.api_error = False
        self._ticker = ticker.upper()
        # start = datetime(1950, 1, 1)
        end = datetime.today()

        try:
            # Reading data from pandas datareader
            self.stock_df = get_data_yahoo(self.ticker,  end = end)

            # Writing the data into a CSV, in case API read fails
            file = StringIO()
            self.stock_df.to_csv(file)
            file.seek(0)
            if exists(join(getcwd(), 'Datasets', f'{self.ticker}.csv')):
                remove(join(getcwd(), 'Datasets', f'{self.ticker}.csv'))
            default_storage.save(join(getcwd(), 'Datasets', f'{self.ticker}.csv'), file)
            file.close()

            # file = StringIO()
            # self.stock_df.to_csv(file)
            # file.seek(0)
            # if exists(f'Datasets/{self.ticker}.csv'):
            #     remove(f'Datasets/{self.ticker}.csv')
            # default_storage.save(f'Datasets/{self.ticker}.csv', file)
            # file.close()

            # file = StringIO()
            # self.stock_df.to_csv(file)
            # file.seek(0)
            # try:
            #     temp = open('Datasets/' + self.ticker + '.csv')
            #     temp.close()
            #     from subprocess import run
            #     run(f"del Datasets\\{self.ticker}.csv", shell = True)
            # except:
            #     pass
            # finally:
            #     default_storage.save('Datasets/' + self.ticker + '.csv', file)
            # file.close()

        # Entered stock symbol is not in database
        except RemoteDataError:
            self.remote_data_error = True
            return None

        # When API fails
        except ConnectionError:
            # Reading from saved file, if exists
            try:
                # self.stock_df = pd.read_csv('Datasets/' + self.ticker + '.csv', index_col = 'Date', parse_dates = True)
                self.stock_df = pd.read_csv(join(getcwd(), 'Datasets', f'{self.ticker}.csv'), index_col = 'Date', parse_dates = True)

            # If file is not found
            except FileNotFoundError:
                self.api_error = True
                return None

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

        # Figure layout
        layout = go.Layout(xaxis = {'title': 'Date'},
                           yaxis = {'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker,
                           xaxis_rangeslider_visible = True)

        # Remove title if comparing two stocks
        fig.update_layout(title = None, xaxis_range = (start, end)) if fig else None

        # Return new figure on new stock and updated figure on comparison
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
        self.signals[f'Short ({short_window} days)'] = self.stock_df['Close'].rolling(window = short_window, min_periods = 1, center = False).mean()
        self.signals[f'Long ({long_window} days)'] = self.stock_df['Close'].rolling(window = long_window, min_periods = 1, center = False).mean()

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

        # Figure layout
        layout = go.Layout(xaxis = {'title': 'Date'},
                           yaxis = {'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker,
                           xaxis_rangeslider_visible = True)

        # Return SMA-CS figure
        return go.Figure(data = [short_avg, long_avg, buy_signal, sell_signal], layout = layout)

    # Backtesting
    # Activate this function only after SMA_CS has been executed
    def backtest(self, initial_capital = 1000000, shares = 100):
        # Dataframe 'portfolio' to backtest SMA-CS
        self.portfolio = pd.DataFrame(index = self.signals.index)
        # try:
        #     self.portfolio = pd.DataFrame(index = self.signals.index)
        # except AttributeError:
        #     return HttpResponse('First implement the crossover strategy')

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

        # Figure layout
        layout = go.Layout(xaxis = {'title': 'Date'},
                           yaxis = {'title': 'Price in $'},
                           hovermode = 'x', title = self.ticker,
                           xaxis_rangeslider_visible = True)

        # Return backtest figure
        return go.Figure(data = [total, buy_signal, sell_signal], layout = layout)

    # Summary of the stock
    @property
    def summary(self):
        previous_close = format(self.stock_df.iloc[-2]['Close'], '.2f')
        today_open = format(self.stock_df.iloc[-1]['Open'], '.2f')
        all_time_max = format(self.stock_df['Adj Close'].max(), '.2f')
        all_time_min = format(self.stock_df['Adj Close'].min(), '.2f')
        today_pct_change = format(self.stock_df.iloc[-2:].pct_change().iloc[-1]['Adj Close'], '.3f')
        monthly_pct_change = format(self.stock_df.resample('M').mean().iloc[-2:].pct_change().iloc[-1]['Adj Close'], '.3f')
        latest_diff = format(self.stock_df.iloc[-2:].diff()['Close'][1], '.2f')
        latest_diff_pct = format(self.stock_df.iloc[-2:].diff()['Close'][1] / self.stock_df.iloc[-2]['Close'] * 100, '.2f')

        return {'ticker': self.ticker,
                'latest_diff': latest_diff,
                'latest_diff_pct': latest_diff_pct,
                'color': 'green' if float(latest_diff) > 0 else 'red',
                'Previous_Close': previous_close,
                "Todays_Open": today_open,
                'All_Time_Maximum': all_time_max,
                'All_Time_Minimum': all_time_min,
                "Todays_Change": today_pct_change,
                'Monthly_Change': monthly_pct_change}
    

# Function when accessing hovermode
def onHome(request):
    # Clearing active stocks and HTTP references on home
    active_stocks.clear()
    http_referer.clear()

    # Collecting all unreferenced objects, if any, using garbage collector
    collect()

    return render(request, 'index.html', {'alerts': {'1': 'Congratulations!', '2': 'For being on the best stock site.'}})

@never_cache
# Function when searching for stocks
def onSubmit(request):
    global http_referer

    # Searched ticker
    ticker = request.GET.get('text', None)

    # HTTP reference of the page
    http_referer.append((request.META['HTTP_REFERER'], ticker))

    # Auto submit without any input
    if ticker is None:
        return HttpResponse('ERROR!')

    # If nothing is passes as input
    elif ticker == '':
        return render(request, 'index.html', {'alerts': {'1': 'You entered nothing!', '2': 'Please enter a valid stock symbol to visualize it.'}})

    # When one stock is compared and back button is pressed
    active_stocks.clear()
    collect()
    ticker=ticker.upper()


    active_stocks[ticker] = StockData(ticker)
    
    # Entered stock symbol is not in database
    if active_stocks[ticker].remote_data_error:
        active_stocks.clear()
        collect()
        return render(request, 'index.html', {'alerts': {'1': f'{ticker} is not a vaild stock symbol!', '2': 'Please enter a vaild one.'}})        

    # API error occurred and backup file is also not present
    elif active_stocks[ticker].api_error:
        active_stocks.clear()
        collect()
        return render(request, 'index.html', {'alerts': {'1': 'Error!', '2': f"API couldn't find {ticker}"}})
        # return HttpResponse(active_stocks[ticker])
    active_stocks['fig'] = active_stocks[ticker].plotClosingPrice()

    plot(active_stocks['fig'], filename = 'static/single.html', auto_open = False)
    # return render(request, 'Chart.html', {'summary': active_stocks[ticker].summary()})
    return render(request, 'Chart.html', {'summary': active_stocks[ticker].summary, 'alert': {}})    # Summary function in progress

@never_cache
# Function when comparing stocks
def onCompare(request):
    global http_referer

    # Ticker to be compared to
    ticker = request.GET.get('text', None)

    # HTTP reference of the page
    http_referer.append((request.META['HTTP_REFERER'], ticker))

    # Auto submit without any input
    if ticker is None:
        return HttpResponse('ERROR!')

    # If nothing is passed as input
    elif ticker == '':
        return render(request, 'Compare.html', {'tickers': (ticker for ticker in active_stocks if ticker != 'fig'), 'original':tuple(active_stocks)[0],
            'alert': 'You entered nothing! Please enter a valid stock symbol to visualize it.'}) if len(active_stocks) > 2 \
            else render(request, 'Chart.html', {'alerts': 'You entered nothing! Please enter a valid stock symbol to visualize it.'})

    ticker=ticker.upper()
    # Adding to active stocks
    if ticker not in active_stocks:
        active_stocks[ticker] = StockData(ticker)

        # Enter stock symbol is not in database
        if active_stocks[ticker].remote_data_error:
            return render(request, 'Compare.html', {'ticker': (ticker for ticker in active_stocks if ticker != 'fig'),
                'alert': f'{ticker} is not a vaild stock symbol! Please enter a vaild one.'}) if len(active_stocks) > 2 \
                else render(request, 'Chart.html', {'alerts': f'{ticker} is not a vaild stock symbol! Please enter a vaild one.'})

        # API error occurred and backup file is also not present
        elif active_stocks[ticker].api_error:
            return render(request, 'Compare.html', {'ticker': (ticker for ticker in active_stocks if ticker != 'fig'),
                'alert': active_stocks[ticker]}) if len(active_stocks) > 2 \
                else render(request, 'Chart.html', {'alerts': {'1': 'ERROR!', '2': f"API couldn't find {ticker}"}})
    
    # Stock symbol already in comparison
    else:
        # Detect whether refresh has happened or not
        if http_referer[-1] == http_referer[-2]:
            http_referer.pop()
            return render(request, 'Compare.html', {'tickers': (ticker for ticker in active_stocks if ticker != 'fig'), 'original':tuple(active_stocks)[0], 'alert': ''})

        # Detect whether back click has happened or not
        try:
            if http_referer[-1] == http_referer[-3]:
                http_referer = http_referer[:0]
                # extra_tickers = tuple(active_stocks)[2:]
                org_ticker = tuple(active_stocks)[0]
                # for ticker in extra_tickers:
                #     active_stocks.pop(ticker)
                # active_stocks['fig'].data = tuple(filter(lambda stock: stock.name == org_ticker, active_stocks['fig'].data))
                return HttpResponseRedirect(f'/submit/?text={org_ticker}')
        # Clicking back twice (GOOG is compared with GOOG -> ERROR)
        except IndexError:
            if len(active_stocks) > 2:
                return HttpResponseRedirect('/home/')

        # Removing first occurrence of ticker to get back  clickfunctionality done properly
        for index, ref in enumerate(http_referer):
            if ref[1] == ticker:
                http_referer.pop(index)
                break

        return render(request, 'Compare.html', {'tickers': (ticker for ticker in active_stocks if ticker != 'fig'), 'original':tuple(active_stocks)[0], 'alert': f'{ticker} is already in comparison.'}) \
            if len(active_stocks) > 2 else render(request, 'Chart.html', {'summary': active_stocks[ticker].summary, 'alerts': f"Can't compare {ticker} with itself!"})

    # Determining best date range
    start = max(active_stocks[ticker].start_date for ticker in active_stocks if ticker != 'fig')
    end = min(active_stocks[ticker].end_date for ticker in active_stocks if ticker != 'fig')
    # print(start, end)
    # Adding ticker trace to current figure
    active_stocks['fig'] = active_stocks[ticker].plotClosingPrice(start = start, end = end,
                                                                  fig = active_stocks['fig'])

    plot(active_stocks['fig'], filename = 'static/multiple.html', auto_open = False)
    return render(request, 'Compare.html', {'tickers': (ticker for ticker in active_stocks if ticker != 'fig'), 'original':tuple(active_stocks)[0], 'alert': ''})

# Function when clicking on trending stocks
def onTrending(request):
    # Get ticker from image
    ticker = tuple(request.GET.keys())[0].split('.')[0]
    
    return HttpResponseRedirect(f'/submit/?text={ticker}')

# Function for Simple Moving Average Crossover Strategy
def onSMA(request):
    # Slow (long) and fast (short) moving windows
    short_window = request.GET.get('short_window', 40)
    long_window = request.GET.get('long_window', 100)
    
    short_window, long_window = int(short_window), int(long_window)
    
    if not short_window or not long_window:
        short_window = 40
        long_window = 100

    for key in active_stocks:
        if key != 'fig':
            plot(active_stocks[key].SMA_CS(short_window = short_window, long_window = long_window),
                filename = 'static/sma.html', auto_open = False)
            break

    return render(request, 'SMACS.html')

# Function on backtest
def onBacktest(request):
    # Taking initial capital and shares as input
    initial_capital = request.GET.get('initial_capital', 1000000)
    shares = request.GET.get('shares', 100)

    if not initial_capital or not shares:
        initial_capital = 1000000
        shares = 100

    for key in active_stocks:
        if key != 'fig':
            plot(active_stocks[key].backtest(initial_capital = initial_capital, shares = shares),
                filename = 'static/back.html', auto_open = False)
            break

    return render(request, 'Backtest.html')

# Function when removing SMA option
def onRemoveSMA(request):
    # plot(active_stocks['fig'], filename = 'static/Page.html', auto_open = False)
    return render(request, 'Chart.html', {'alerts': ''})

# Function when removing comparison stock
def onRemoveComparison(request):
    # Ticker to remove
    ticker = tuple(request.GET.keys())[0]

    # Removing from active stocks and figure as well
    try:
        active_stocks.pop(ticker)
        collect()
        active_stocks['fig'].data = tuple(filter(lambda stock: stock.name != ticker, active_stocks['fig'].data))

        # Plot new figure without removed comparison stock
        plot(active_stocks['fig'], filename = 'static/multiple.html', auto_open = False)
    except:
        return render(request, 'Compare.html', {'tickers': active_stocks, 'original':tuple(active_stocks)[0], 'alert':''})

    return render(request, 'Compare.html', {'tickers': (ticker for ticker in active_stocks if ticker != 'fig'), 'original':tuple(active_stocks)[0], 'alert': ''}) if len(active_stocks['fig'].data) > 1 else render(request, 'Chart.html', {'summary': active_stocks[tuple(active_stocks)[0]].summary,'alerts': ''})

# Currently active stocks and figure
active_stocks = {}
http_referer = []
