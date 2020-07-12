#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-04-05'
__email__ = 'fredflorescfa@gmail.com'

import numpy as np
from scipy.stats import kurtosis, skew, norm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn
from utilities import util

seaborn.set()


def get_patch(patches, search_value):
    """Get the index of the histogram bin that corresponds to search value."""
    min_distance = float("inf")  # initialize min_distance with infinity
    patch_index = 0
    for i, rectangle in enumerate(patches):  # iterate over every bar
        tmp = abs(  # tmp = distance from middle of the bar to search_value
            (rectangle.get_x() +
             (rectangle.get_width() * (1 / 2))) - search_value)
        if tmp < min_distance:  # we are searching for the bar with x cordinate
            # closest to search_value
            min_distance = tmp
            patch_index = i

    return patch_index


def histogram(s, title='My Data', bins=100, figsize=(12, 12)):

    y = util.numpy_array(s)
    is_valid = ~np.isnan(y)

    plt.figure(figsize=figsize)
    bin_y, bin_x, patches = plt.hist(y[is_valid], bins=bins, density=True, color='darkseagreen')
    for i, rectangle in enumerate(patches):
        patches[i].set_edgecolor('darkseagreen')  # overriding seaborn white edge

    plt.xlim(np.min(y[is_valid]), np.max(y[is_valid]))
    plt.ylabel('Probability Density')
    plt.title(title)

    # Identify median in histogram
    p10 = np.quantile(y[is_valid], 0.10)
    p25 = np.quantile(y[is_valid], 0.25)
    p50 = np.quantile(y[is_valid], 0.50)
    p75 = np.quantile(y[is_valid], 0.75)
    p90 = np.quantile(y[is_valid], 0.90)

    idx = get_patch(patches, p50)
    patches[idx].set_facecolor('goldenrod')
    patches[idx].set_edgecolor('goldenrod')
    patches[idx].set_label('Median')

    # Identify mean in histogram
    mu = np.mean(y[is_valid])
    sigma = np.std(y[is_valid])

    idx = get_patch(patches, mu)
    patches[idx].set_facecolor('darkred')
    patches[idx].set_edgecolor('darkred')
    patches[idx].set_label('Mean, +/- Std Dev')

    idx = get_patch(patches, mu + sigma)
    patches[idx].set_facecolor('darkred')
    patches[idx].set_edgecolor('darkred')

    idx = get_patch(patches, mu - sigma)
    patches[idx].set_facecolor('darkred')
    patches[idx].set_edgecolor('darkred')

    # Print statistics in right margin
    right_margin = 0.76
    plt.subplots_adjust(right=right_margin)
    text_left = right_margin + .01

    kurt = kurtosis(y[is_valid])
    sku = skew(y[is_valid])
    stats = {r'Total = {:.0f}': (len(y), 0.85, 'black'),
             r'Valid = {:.0f}': (np.sum(is_valid), 0.82, 'black'),
             r'Nan = {:.0f}': (np.sum(~is_valid), 0.79, 'black'),
             r'Mean = {:.3f}': (mu, 0.73, 'black'),
             r'Stdev = {:.3f}': (sigma, 0.70, 'black'),
             r'Kurtosis = {:.3f}': (kurt, 0.64, 'black'),
             r'Skew = {:.3f}': (sku, 0.61, 'black'),
             r'P25 = {:.3f}': (p25, 0.55, 'black'),
             r'Median = {:.3f}': (p50, 0.52, 'black'),
             r'P75 = {:.3f}': (p75, 0.49, 'black'),
             r'IQR = {:.3f}': (p75-p25, 0.43, 'black'),
             r'IDR = {:.3f}': (p90-p10, 0.40, 'black')}
    
    for key in stats.keys():
        plt.gcf().text(text_left, stats[key][1], key.format(stats[key][0]), color=stats[key][2])

    # Superimpose normal distribution curve
    x = np.linspace(norm.ppf(0.001, loc=mu, scale=sigma), norm.ppf(0.999, loc=mu, scale=sigma), 100)
    plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'r-', lw=3, color='grey', alpha=0.6, label='Normal')

    plt.legend(fontsize=10)

    plt.show()


def histogram_new(s, title='My Data', bins=100, figsize=(12, 8), winsor_bound=[.01, 0.99],
                  xlim=None, ylim=None, axvline=None, show_normal=False):
    """ Parameterize winsorization and remove mean, stdev colored bars. """
    data_types = {pd.core.series.Series: (lambda x: x.values),
                  np.ndarray: (lambda x: x),
                  list: (lambda x: np.array(x))}

    assert type(s) in data_types.keys(), 'invalid data type. Enter numpy array, pandas series , or list of float.'

    y = data_types[type(s)](s)
    is_valid = ~np.isnan(y)

    y = y[is_valid]
    lo_win = np.nanquantile(y, winsor_bound[0])
    hi_win = np.nanquantile(y, winsor_bound[1])

    y[y < lo_win] = lo_win
    y[y > hi_win] = hi_win

    plt.figure(figsize=figsize)
    bin_y, bin_x, patches = plt.hist(y, bins=bins, density=True, color='darkseagreen')
    for i, rectangle in enumerate(patches):
        patches[i].set_edgecolor('darkseagreen')  # overriding seaborn white edge

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    if axvline is not None:
        plt.axvline(axvline, color='darkred', lw=1, ls='--')

    plt.ylabel('Probability Density')
    plt.title(title)

    # Identify median in histogram
    p10 = np.quantile(y, 0.10)
    p25 = np.quantile(y, 0.25)
    p50 = np.quantile(y, 0.50)
    p75 = np.quantile(y, 0.75)
    p90 = np.quantile(y, 0.90)

    mu = np.mean(y)
    sigma = np.std(y)

    # Print statistics in right margin
    right_margin = 0.76
    plt.subplots_adjust(right=right_margin)
    text_left = right_margin + .03

    kurt = kurtosis(y)
    sku = skew(y)
    stats = {r'Total = {:.0f}': (len(y), 0.85, 'black'),
             r'Valid = {:.0f}': (np.sum(is_valid), 0.82, 'black'),
             r'Nan = {:.0f}': (np.sum(~is_valid), 0.79, 'black'),
             r'Mean = {:.3f}': (mu, 0.73, 'black'),
             r'Stdev = {:.3f}': (sigma, 0.70, 'black'),
             r'Kurtosis = {:.3f}': (kurt, 0.64, 'black'),
             r'Skew = {:.3f}': (sku, 0.61, 'black'),
             r'P25 = {:.3f}': (p25, 0.55, 'black'),
             r'Median = {:.3f}': (p50, 0.52, 'black'),
             r'P75 = {:.3f}': (p75, 0.49, 'black'),
             r'IQR = {:.3f}': (p75 - p25, 0.43, 'black'),
             r'IDR = {:.3f}': (p90 - p10, 0.40, 'black')}

    for key in stats.keys():
        plt.gcf().text(text_left, stats[key][1], key.format(stats[key][0]), color=stats[key][2], fontsize=10)

    # Superimpose normal distribution curve
    if show_normal:
        x = np.linspace(norm.ppf(0.001, loc=mu, scale=sigma), norm.ppf(0.999, loc=mu, scale=sigma), 100)
        plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'r-', lw=3, color='grey', alpha=0.6,
                 label='Normal Distribution')
        plt.legend(fontsize=10)
        r = norm.rvs(size=1000)
    plt.show()


def series(df, title='My Chart', colors=seaborn.color_palette(n_colors=12), figsize=(6.4, 4.8),
           digits=2):
    """ Plot line(s) chart from a Pandas Series or DataFrame.
        Assumes each column is a series and the index represents the order."""

    data_types = [pd.core.frame.DataFrame, pd.core.series.Series]

    assert type(df) in data_types, 'invalid data type. Enter Pandas Series or DataFrame.'
    assert df.shape[1] <= len(colors), 'There are not enough colors.'

    df = df.sort_index(ascending=True)

    fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    fig.suptitle(title)

    lines = axs[0].plot(df.index, df)
    sig_digits = '%.' + str(digits) + 'f'
    axs[0].yaxis.set_major_formatter(FormatStrFormatter(sig_digits))

    medianprops = dict(linestyle='-', linewidth=1, color='black')
    boxes = axs[1].boxplot(df.T, labels=df.columns, patch_artist=True, medianprops=medianprops,
                           showfliers=False)

    for idx in np.arange(0, df.shape[1]):
        lines[idx].set_label(df.columns[idx])
        lines[idx].set_color(colors[idx])

    for patch, c in zip(boxes['boxes'], colors):
        patch.set_facecolor(c)

    for patch, c in zip(boxes['fliers'], colors):
        patch.set_color(c)

    # med = df.median().values
    for line in boxes['medians']:
        x, y = line.get_xydata()[1]  # top of median line
        axs[1].text(x, y, sig_digits % y, horizontalalignment='left', fontsize=10, rotation=45)

    plt.show()


def bar(x, figsize=(16,6), title=('title', 14), xlabel=('xlabel', 12), ylabel=('ylabel', 12),
        legend_items=np.arange(0,20), legend_title=('leg_title', 12), legend_out=True, ylim=None,
        sig_digits=2, width=0.4, xticklabels=None, rotation=0, colors=seaborn.color_palette(n_colors=10)):

    """ Plot each row of x as a bar chart series.
            row values = bar heights (y-axis)
            row index = series in legend
            column index = categories on x-axis"""

    assert isinstance(x, np.ndarray), 'Invalid numpy array'

    rows = np.arange(1, x.shape[0] + 1)
    columns = np.arange(x.shape[1])

    for arg in [title, xlabel, ylabel, legend_title]:
        assert isinstance(arg, tuple), 'Invalid tuple'
        assert isinstance(arg[0], str), 'Invalid string in tuple'
        assert isinstance(arg[1], int), 'Invalid integer for fontsize'

    if xticklabels is not None:
        assert isinstance(xticklabels, dict), 'Invalid dictionary for xticklabels.'
        for idx in columns:
            assert idx in xticklabels.keys(), str(x) + ' key not found in xticklabels dictionary.'
    else:
        xticklabels = {d: d for d in columns}

    def auto_label(rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(('%.' + str(sig_digits) + 'f') % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=14)

    fig, ax = plt.subplots(figsize=figsize)

    centers = np.arange(-(len(rows)-1), len(rows), 2)

    for pos, ctr in zip(rows, centers):
        rects = ax.bar(columns + (ctr * (width/2)), x[pos-1], width, label=str(pos), facecolor=colors[pos-1],
                       edgecolor=colors[pos-1])
        auto_label(rects, ax)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel[0], fontsize=ylabel[1])
    ax.set_xlabel(xlabel[0], fontsize=xlabel[1])
    ax.set_title(title[0], fontsize=title[1])
    ax.set_xticks(columns)
    ax.set_xticklabels([xticklabels[d] for d in columns], rotation=rotation)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if legend_out:
        ax.legend(legend_items, title=legend_title[0], fontsize=legend_title[1],
                  bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    else:
        ax.legend(legend_items, title=legend_title[0], fontsize=legend_title[1])

    fig.tight_layout()

    plt.show()


def grid_2d(X, animated=False, cmap='RdYlGn'):
    """Display grid of numpy array X where each cell is filled with a color based on its value."""

    # make a color map of fixed colors
    # cmap = mpl.colors.ListedColormap(['white', 'black'])
    # bounds = [-6, -2, 2, 6]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    # img = plt.imshow(X, interpolation='nearest', cmap=cmap, norm=norm)
    img = plt.imshow(X, interpolation='nearest', cmap=cmap, animated=animated)
    img.axes.grid(False)

    # make a color bar
    #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-5, 0, 5])

    plt.show()


def candlestick(df, close, high, low, title='', figsize=(14, 6)):
    df = df.sort_index(ascending=True)
    plt.figure(figsize=figsize)
    plt.plot(df.index, df[close], '.', c='darkblue')
    plt.plot((df.index, df.index), (df[high], df[low]), c='indianred', lw=1)
    plt.ylabel('Price per Share', fontsize=14)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

