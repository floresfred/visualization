#!/usr/bin/env python

__author__ = 'Fred Flores'
__version__ = '0.0.1'
__date__ = '2020-04-05'
__email__ = 'fredflorescfa@gmail.com'

import numpy as np
from scipy.stats import kurtosis, skew, norm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn

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


def histogram(s, title='My Data', bins=100):

    data_types = {pd.core.series.Series: (lambda x: x.values),
                  np.ndarray: (lambda x: x),
                  list: (lambda x: np.array(x))}

    assert type(s) in data_types.keys(), 'invalid data type. Enter numpy array, pandas series , or list of float.'

    y = data_types[type(s)](s)
    is_valid = ~np.isnan(y)

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