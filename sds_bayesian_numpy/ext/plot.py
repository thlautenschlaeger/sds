from sds_bayesian_numpy.ext.utils import series_to_windows
import matplotlib.pyplot as plt
import numpy as np
from hips.plotting.colormaps import gradient_cmap
import seaborn as sns


def plot_series(series, title='Stacked plots', names=None, **kwargs):
    """ Plots multiple series """
    num_plots = series.shape[1]
    x = np.arange(series.shape[0])
    plot_names = True if names != None and len(names) == num_plots else False

    fig, axs = plt.subplots(num_plots)

    for n in range(num_plots):
        axs[n].plot(x, series[:, n])

        if plot_names:
            axs[n].set_title(names[n], fontsize=8)

    plt.show()


def plot_series_prediction(true_series, predicted_series, std=None, names=None, title=None, indicator_pos=None):
    num_plots = true_series.shape[1]
    x = np.arange(true_series.shape[0])
    plot_names = True if names != None and len(names) == num_plots else False

    fig, axs = plt.subplots(num_plots)

    fig.suptitle(title)

    if std is None:
        std = np.zeros((true_series.shape[0], num_plots))

    if num_plots == 1:
        axs.plot(x, true_series, label='true')
        axs.plot(x, predicted_series, label='prediction')
        axs.fill_between(x, predicted_series + std, predicted_series - std)

        if indicator_pos is not None:
            xs = indicator_pos * np.ones(1, true_series.shape[0])
            axs.plot(xs, true_series, '.')

        if plot_names:
            axs.set_title(names[0], fontsize=8)
    else:
        for n in range(num_plots):
            axs[n].plot(x, true_series[:, n], label='true')
            axs[n].plot(x, predicted_series[:, n], label='prediction')
            axs[n].fill_between(x, predicted_series[:, n] - std[:, n], predicted_series[:, n] + std[:, n], alpha=0.4, color='orange')

            if indicator_pos is not None:
                axs[n].axvline(x=indicator_pos, linestyle=':', c='gray', linewidth=0.8)

            if plot_names:
                axs[n].set_title(names[n], fontsize=8)

    plt.legend()

    return fig


def plot_viterbi(seq1, seq2):
    """ visualizes state transitions in a plot of two sequences """
    sns.set()
    color_names = ["windows blue", "red", "amber", "faded green",
                   "dusty purple", "orange", "pale red", "medium green",
                   "denim blue", "muted purple"]
    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.imshow(seq1[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # plt.xlim(0, len(x[_seq]))
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(seq2[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    # plt.xlim(0, len(x[_seq]))
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()

def plot_viterbi_single(seq):
    """ visualizes state transitions in a plot of two sequences """
    # sns.set()
    color_names = ["windows blue", "red", "amber", "faded green",
                   "dusty purple", "orange", "pale red", "medium green",
                   "denim blue", "muted purple"]
    colors = sns.xkcd_palette(color_names)
    cmap = gradient_cmap(colors)

    plt.figure(figsize=(8, 2))
    plt.subplot(111)
    plt.imshow(seq[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])
    # plt.xlim(left=-5, right=len(seq[0]) + 5)

    plt.xlabel("step")

    plt.tight_layout()
    plt.show()




# _obs = np.loadtxt('vbhmm/data/xploras/returns.txt')
names = ['BTC', 'XRP', 'LTC', 'ETH', 'USDT', 'LOL']

# series = series_to_windows(_obs, 50)
# truth = series[0]
# prediction = series[1]

# plot_series_prediction(truth, prediction, names)
# plot_series(series[0], names=names)


