from math import pi
import pandas as pd
from alpha_vantage_data import AlphaVantage
import matplotlib.pyplot as plt
from visualization import viz

from bokeh.plotting import figure, show, output_file

av = AlphaVantage.AlphaVantage(key='C1MLXDST5B1WH97M')

data_dict = av.get_intraday('googl')['Time Series (1min)']
df = pd.DataFrame(data_dict).T
df = df.astype(float)
df = df.reset_index()
df['index'] = pd.to_datetime(df['index'], format='%Y-%m-%d %H:%M:%S')

df = df.rename(columns={'index': 'date_time',
                        '1. open': 'open',
                        '2. high': 'high',
                        '3. low': 'low',
                        '4. close': 'close',
                        '5. volume': 'volume'})

mids = (df['open'] + df['close'])/2
spans = abs(df['close']-df['open'])

inc = df['close'] > df['open']
dec = df['open'] > df['close']
w = 25000  # half day in ms

output_file("candlestick.html", title="candlestick.py example")

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, toolbar_location="left",
           title="Apple Stock Price")

p.segment(df['date_time'], df['high'], df['date_time'], df['low'], color="black")
p.rect(df['date_time'][inc], mids[inc], w, spans[inc], fill_color="#D5E1DD", line_color="black")
p.rect(df['date_time'][dec], mids[dec], w, spans[dec], fill_color="#F2583E", line_color="black")

# p.title = "MSFT Candlestick"
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha = 0.3

show(p)  # open a browser