import mne
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch
import matplotlib.colors as mcolors
from matplotlib import ticker


# High-level

def plot_sensor_data(ax, data, raw, xvect=None, lw=0.5,
                     xticks=None, xticklabels=None,
                     sensor_cols=True, base=1, xtick_skip=1):
    if xvect is None:
        xvect = np.arange(obs.shape[0])
    fx, xticklabels, xticks = prep_scaled_freq(base, xvect)

    if sensor_cols:
        colors, pos, outlines = get_mne_sensor_cols(raw)
    else:
        colors = None

    plot_with_cols(ax, data, fx, colors, lw=lw)
    ax.set_xlim(fx[0], fx[-1])

    if xticks is not None:
        ax.set_xticks(xticks[::xtick_skip])
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels[::xtick_skip])


def plot_sensor_spectrum(ax, psd, raw, xvect, sensor_proj=False,
                         xticks=None, xticklabels=None, lw=0.5,
                         sensor_cols=True, base=1, ylabel=None, xtick_skip=1):

    plot_sensor_data(ax, psd, raw, base=base, sensor_cols=sensor_cols, lw=lw,
                     xvect=xvect, xticks=xticks, xticklabels=xticklabels, xtick_skip=xtick_skip)
    decorate_spectrum(ax, ylabel=ylabel)
    ax.set_ylim(psd.min())

    if sensor_proj:
        axins = ax.inset_axes([0.6, 0.6, 0.37, 0.37])
        plot_channel_layout(axins, raw)


def subpanel_label(ax, label, xf=-0.1, yf=1.1, ha='center'):
    ypos = ax.get_ylim()[0]
    yyrange = np.diff(ax.get_ylim())[0]
    ypos = (yyrange * yf) + ypos
    # Compute letter position as proportion of full xrange.
    xpos = ax.get_xlim()[0]
    xxrange = np.diff(ax.get_xlim())[0]
    xpos = (xxrange * xf) + xpos
    ax.text(xpos, ypos, label, horizontalalignment=ha,
            verticalalignment='center', fontsize=20, fontweight='bold')

# Helpers

def decorate_spectrum(ax, ylabel='Power'):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(ylabel)


def decorate_timseries(ax, ylabel='Power'):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(ylabel)


def plot_with_cols(ax, data, xvect, cols=None, lw=0.5):
    if cols is not None:
        for ii in range(data.shape[1]):
            ax.plot(xvect, data[:, ii], lw=lw, color=cols[ii, :])
    else:
        ax.plot(xvect, data, lw=lw)


def prep_scaled_freq(base, freq_vect):
    """Assuming ephy freq ranges for now - around 1-40Hz"""
    fx = freq_vect**base
    if base < 1:
        nticks = int(np.floor(np.sqrt(freq_vect[-1])))
        #ftick = np.array([2**ii for ii in range(6)])
        ftick = np.array([ii**2 for ii in range(1,nticks+1)])
        ftickscaled = ftick**base
    else:
        # Stick with automatic scales
        ftick = None
        ftickscaled = None
    return fx, ftick, ftickscaled


# MNE Helpers


def get_mne_sensor_cols2(info):

    chs = [info['chs'][i] for i in range(len(info['chs']))]
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    x, y, z = locs3d.T
    colors = mne.viz.evoked._rgb(x, y, z)
    pos, outlines = mne.viz.evoked._get_pos_outlines(info,
                                                     range(len(info['chs'])),
                                                     sphere=None)

    return colors, pos, outlines


def plot_channel_layout2(ax, info, size=30, marker='o'):

    ax.set_adjustable('box')
    ax.set_aspect('equal')

    colors, pos, outlines = get_mne_sensor_cols2(info)
    pos_x, pos_y = pos.T
    mne.viz.evoked._prepare_topomap(pos, ax, check_nonzero=False)
    ax.scatter(pos_x, pos_y,
               color=colors, s=size * .8,
               marker=marker, zorder=1)
    mne.viz.evoked._draw_outlines(ax, outlines)

def get_mne_sensor_cols(raw, picks=None):
    if picks is not None:
        raw.pick_types(**picks)

    chs = [raw.info['chs'][i] for i in range(len(raw.info['chs']))]
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    x, y, z = locs3d.T
    colors = mne.viz.evoked._rgb(x, y, z)
    pos, outlines = mne.viz.evoked._get_pos_outlines(raw.info,
                                                     range(len(raw.info['chs'])),
                                                     sphere=None)

    return colors, pos, outlines


def plot_channel_layout(ax, raw, size=30, marker='o'):

    ax.set_adjustable('box')
    ax.set_aspect('equal')

    colors, pos, outlines = get_mne_sensor_cols(raw)
    pos_x, pos_y = pos.T
    mne.viz.evoked._prepare_topomap(pos, ax, check_nonzero=False)
    ax.scatter(pos_x, pos_y,
               color=colors, s=size * .8,
               marker=marker, zorder=1)
    mne.viz.evoked._draw_outlines(ax, outlines)


def plot_joint_spectrum(ax, psd, raw, xvect, freqs='auto', base=1, topo_scale='joint', lw=0.5, ylabel='Power', title='', ylim=None, xtick_skip=1):

    if ylim is None:
        # Plot invisible lines to get correct xy lims - probably a better way to do
        # this using update_datalim but I can't find it.
        plot_sensor_spectrum(ax, psd, raw, xvect, base=base, lw=0, ylabel=ylabel)
        ylim = ax.get_ylim()
    else:
        ax.plot(xvect, np.linspace(ylim[0], ylim[1], len(xvect)), lw=0)
        ax.set_ylim(*ylim)

    fx, xtl, xt = prep_scaled_freq(base, xvect)

    if freqs == 'auto':
        freqs = signal.find_peaks(psd.mean(axis=1), distance=10)[0]
        if 0 not in freqs:
            freqs = np.r_[0, freqs]
    else:
        # Convert Hz to samples in freq dim
        freqs = [np.argmin(np.abs(xvect - f)) for f in freqs]

    topo_centres = np.linspace(0, 1, len(freqs)+2)[1:-1]
    topo_width = 0.4

    # Shrink axes to make space for topos
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height*0.65])

    shade = [0.7, 0.7, 0.7]

    #if topo_scale is 'joint':
    #    vmin = psd.mean(axis=1).min()
    #    vmax = psd.mean(axis=1).max()
    #    # Set colourmaps
    #    if np.all(np.sign((vmin, vmax))==1):
    #        # Reds if all positive
    #        cmap = 'Reds'
    #    elif np.all(np.sign((vmin, vmax))==-1):
    #        # Blues if all negative
    #        cmap = 'Blues'
    #    elif np.all(np.sign((-vmin, vmax))==1):
    #        # RdBu diverging from zero if split across zero
    #        cmap = 'RdBu_r'
    #else:
    #    vmin = None
    #    vmax = None
    norm = None
    if topo_scale is 'joint':
        vmin = obs.mean(axis=1).min()
        vmax = obs.mean(axis=1).max()
        # Set colourmaps
        if np.all(np.sign((vmin, vmax))==1):
            # Reds if all positive
            cmap = 'Reds'
        elif np.all(np.sign((vmin, vmax))==-1):
            # Blues if all negative
            cmap = 'Blues'
        elif np.all(np.sign((-vmin, vmax))==1):
            # RdBu diverging from zero if split across zero
            cmap = 'RdBu_r'
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
            vmax = np.max((vmin, vmax))
            vmin = -vmax
    else:
        vmin = None
        vmax = None

    for idx in range(len(freqs)):
        # Create topomap axis
        topo_pos = [topo_centres[idx] - 0.2, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos)

        if topo_scale is None:
            vmin = psd[freqs[idx], :].min()
            vmax = psd[freqs[idx], :].max()
            # Set colourmaps
            if np.all(np.sign((vmin, vmax))==1):
                # Reds if all positive
                cmap = 'Reds'
            elif np.all(np.sign((vmin, vmax))==-1):
                # Blues if all negative
                cmap = 'Blues'
            elif np.all(np.sign((-vmin, vmax))==1):
                # RdBu diverging from zero if split across zero
                cmap = 'RdBu_r'

        # Draw topomap itself
        #im, cn = mne.viz.plot_topomap(psd[freqs[idx], :], raw.info, axes=topo,
        #                              cmap=cmap, vmin=vmin, vmax=vmax)
        dat = psd[freqs[idx], :]
        if len(np.unique(np.sign(dat))) == 2:
            print('Crossing')
            dat = dat / np.abs(dat).max()
            im, cn = mne.viz.plot_topomap(dat, raw.info, axes=topo, cmap='RdBu_r',
                                          vlim=(-1, 1), show=False) #vmin=-1, vmax=1, show=False)
        elif np.unique(np.sign(dat)) == [1]:
            print('Positive')
            dat = dat - dat.min()
            dat = dat / dat.max()
            im, cn = mne.viz.plot_topomap(dat, raw.info, axes=topo, cmap='Reds',
                                          vlim=(0, 1), show=False) #vmin=0, vmax=1, show=False)
        elif np.unique(np.sign(dat)) == [-1]:
            print('Negative')
            dat = dat - dat.max()
            dat = -(dat / dat.min())
            im, cn = mne.viz.plot_topomap(-dat, raw.info, axes=topo, cmap='Blues',
                                          vlim=(0, 1), show=False) #vmin=0, vmax=1, show=False)
        print('{} - {}'.format(dat.min(), dat.max()))

        # Add angled connecting line
        xy = (fx[freqs[idx]], ax.get_ylim()[1])
        con = ConnectionPatch(xyA=xy, xyB=(0, topo.get_ylim()[0]),
                              coordsA=ax.transData, coordsB=topo.transData,
                              axesA=ax, axesB=topo, color=shade, lw=2)
        ax.get_figure().add_artist(con)

        #if idx == len(freqs) - 1:
        #    cb_pos = [0.95, 1.2, 0.05, 0.4]
        #    cax = ax.inset_axes(cb_pos)
        #    cb = plt.colorbar(im, cax=cax, boundaries=np.linspace(vmin, vmax))
        #    tks = _get_sensible_ticks(round_to_first_sig(vmin), round_to_first_sig(vmax), 3)
        #    cb.set_ticks([vmin, vmax])
        #    cb.set_ticklabels(['min', 'max'])

    # Add vertical lines
    ax.vlines(fx[freqs], ax.get_ylim()[0], ax.get_ylim()[1], color=shade, lw=2)

    plot_sensor_spectrum(ax, psd, raw, xvect, base=base, lw=lw, ylabel=ylabel, xtick_skip=xtick_skip)
    ax.set_title(title)
    ax.set_ylim(ylim)


def test_base(b):
    plt.figure()
    f = np.linspace(0, 40)
    fs = f**b
    fx = np.array([2**ii for ii in range(6)])

    plt.plot(fs, f)
    plt.xticks(fx**b, fx)


def plot_sensorspace_clusters(dat, P, raw, ax, xvect=None, ylabel='Power', topo_scale='joint', base=1, lw=0.5, title=None, thresh=95):
    from matplotlib.patches import ConnectionPatch
    clu, obs = P.get_sig_clusters(thresh, dat)
    if xvect is None:
        xvect = np.arange(obs.shape[0])

    # Start plotting
    plot_sensor_spectrum(ax, obs, raw, xvect, base=base, lw=lw, ylabel=ylabel)
    fx, xtl, xt = prep_scaled_freq(base, xvect)

    shade = [0.7, 0.7, 0.7]
    xf = -0.03

    # Shrink axes to make space for topos
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width, pos.height*0.65])

    # sort clusters by ascending freq
    forder = np.argsort([c[2][0].mean() for c in clu])
    clu = [clu[c] for c in forder]

    norm = None
    if topo_scale is 'joint':
        vmin = obs.mean(axis=1).min()
        vmax = obs.mean(axis=1).max()
        # Set colourmaps
        if np.all(np.sign((vmin, vmax))==1):
            # Reds if all positive
            cmap = 'Reds'
        elif np.all(np.sign((vmin, vmax))==-1):
            # Blues if all negative
            cmap = 'Blues'
        elif np.all(np.sign((-vmin, vmax))==1):
            # RdBu diverging from zero if split across zero
            cmap = 'RdBu_r'
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
            vmax = np.max((vmin, vmax))
            vmin = -vmax
    else:
        vmin = None
        vmax = None
    print('{} : {} - {}'.format(vmin, vmax, cmap))
    ax.set_title(title)

    if len(clu) == 0:
        # put up an empty axes anyway
        topo_pos = [0.3, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos, frame_on=False)
        topo.set_xticks([])
        topo.set_yticks([])
        return

    if len(clu) >3:
        # Plot topos for three largest clusters
        cstats = [np.abs(c[0]) for c in clu]
        topo_cv = np.argsort(cstats)[-3]
        topo_plot = np.array([True if np.abs(c[0]) >= cstats[topo_cv] else False for c in clu])
    else:
        topo_plot = np.array([True for c in clu])

    topo_centres = np.linspace(0, 1, topo_plot.sum()+2)[1:-1]
    topo_width = 0.4

    stupid_counter = 0
    for c in range(len(clu)):
        inds = np.where(clu==c+1)[0]
        channels = np.zeros((obs.shape[1], ))
        channels[clu[c][2][1]] = 1
        if len(channels) == 204:
            channels = np.logical_and(channels[::2], channels[1::2])
        times = np.zeros((obs.shape[0], ))
        times[clu[c][2][0]] = 1
        tinds = np.where(times)[0]
        #if len(tinds) == 1:
        #    continue
            #tinds = [tinds[0], tinds[0]+1]
        ax.axvspan(fx[tinds[0]], fx[tinds[-1]], facecolor=shade, alpha=0.5)

        if topo_plot[c]:

            topo_pos = [topo_centres[stupid_counter] - 0.2, 1.2, 0.4, 0.4]
            stupid_counter += 1
            topo = ax.inset_axes(topo_pos)
            dat = obs[tinds, :].mean(axis=0)

            # Scale topo by min and max of whole data range.
            #dat = dat / np.abs(dat).max()
            dat = dat.mean() + stats.zscore(dat)
            vmin = dat.min()
            vmax = dat.max()
            vmin = -vmax if vmax > np.abs(vmin) else vmin
            vmax = -vmin if np.abs(vmin) > vmax else vmax
            im, cn = mne.viz.plot_topomap(dat, raw.info, axes=topo, cmap='RdBu_r',
                                        #vmin=vmin, vmax=vmax, mask=channels.astype(int),
                                        vlim=(vmin, vmax), mask=channels.astype(int),
                                        show=False)

            #if len(np.unique(np.sign(dat))) == 2:
            #    dat = dat / np.abs(dat).max()
            #    print('Crossing {} - {}'.format(dat.min(), dat.max()))
            #    im, cn = mne.viz.plot_topomap(dat, raw.info, axes=topo, cmap='RdBu_r',
            #                                  vmin=dat.min(), vmax=dat.max(), mask=channels.astype(int),
            #                                  show=False)
            #elif np.unique(np.sign(dat)) == [1]:
            #    dat = dat - dat.min()
            #    dat = dat / dat.max()
            #    print('Positive {} - {}'.format(dat.min(), dat.max()))
            #    im, cn = mne.viz.plot_topomap(dat, raw.info, axes=topo, cmap='Reds',
            #                                  vmin=0, vmax=1, mask=channels.astype(int),
            #                                  show=False)
            #elif np.unique(np.sign(dat)) == [-1]:
            #    dat = dat - dat.max()
            #    dat = dat / dat.min()
            #    print('Positive {} - {}'.format(dat.min(), dat.max()))
            #    im, cn = mne.viz.plot_topomap(dat, raw.info, axes=topo, cmap='Blues_r',
            #                                  vmin=-1, vmax=0, mask=channels.astype(int),
            #                                  show=False)

            #if norm is not None:
            #    im.set_norm(norm)

            xy = (fx[tinds].mean(), ax.get_ylim()[1])
            con = ConnectionPatch(xyA=xy, xyB=(0, topo.get_ylim()[0]),
                                coordsA=ax.transData, coordsB=topo.transData,
                                axesA=ax, axesB=topo, color=shade)
            plt.gcf().add_artist(con)

        if c == len(clu) - 1:
            cb_pos = [0.95, 1.2, 0.05, 0.4]
            cax = ax.inset_axes(cb_pos)
            #cb = plt.colorbar(im, cax=cax, boundaries=np.linspace(vmin, vmax))
            #tks = _get_sensible_ticks(round_to_first_sig(vmin), round_to_first_sig(vmax), 3)
            #cb.set_ticks(tks)
            cb = plt.colorbar(im, cax=cax) #, boundaries=np.linspace(-1, 1))
            #tks = _get_sensible_ticks(round_to_first_sig(vmin), round_to_first_sig(vmax), 3)
            cb.set_ticks([vmin, 0, vmax], ['min', '0', 'max'])
            cb.set_ticklabels(['min', '0', 'max'])



def _get_sensible_ticks(vmin, vmax, nbins=3):
    """Return sensibly rounded tick positions based on a plotting range.

    Based on code in matplotlib.ticker
    Assuming symmetrical axes and 3 ticks for the moment

    """
    scale, offset = ticker.scale_range(vmin, vmax)
    if vmax/scale > 0.5:
        scale = scale / 2
    edge = ticker._Edge_integer(scale, offset)
    low = edge.ge(vmin)
    high = edge.le(vmax)

    ticks = np.linspace(low, high, nbins) * scale

    return ticks

def round_to_first_sig(x):
  return np.round(x, -int(np.floor(np.log10(np.abs(x)))))


def vrange_logic(data):

    # If all positive, use Reds

    # If all negative, use Blues

    # If split, use white in middle RdBu
    return None
