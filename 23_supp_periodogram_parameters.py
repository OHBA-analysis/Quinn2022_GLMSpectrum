"""This script produces a figure summarising periodogram parameter options."""

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import sails

import lemon_plotting
from glm_config import cfg
from lemon_support import get_eeg_data

#%% ----------------------------------------------------------
# GLM-Prep

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}_preproc_raw.fif')
st = osl.utils.Study(fbase)

fname = st.get(subj='sub-010060')[0]

runname = fname.split('/')[-1].split('.')[0]
print('processing : {0}'.format(runname))

subj_id = osl.utils.find_run_id(fname)

raw = mne.io.read_raw_fif(fname, preload=True)
refraw = raw.copy().pick_types(eeg=True)

# Get data
XX = get_eeg_data(raw).T
print(XX.shape)
fs = raw.info['sfreq']

#%% ---------------------------------------------------------
# Explore window length

wls = [25, 50, 250, 1000, 2500]
win_len = []
for nperseg in wls:
    f, pxx = sails.stft.periodogram(XX, axis=0, nperseg=nperseg, mode='magnitude', fs=fs)
    win_len.append((f, pxx))

srs = [25, 50, 100, 250, 500]
sample_rate = []
for fs in srs:
    YY = sails.utils.fast_resample(XX, 250/fs, axis=0)
    f, pxx = sails.stft.periodogram(YY, axis=0, nperseg=250, mode='magnitude', fs=fs)
    sample_rate.append((f, pxx))
fs = raw.info['sfreq']

dls = [1, 3, 5, 30, 60]
data_len = []
for dl in dls:
    f, pxx = sails.stft.periodogram(XX[:int(dl*fs), :], axis=0,
                                    nperseg=250, mode='magnitude', fs=fs)
    data_len.append((f, pxx))


sfreqs = np.arange(100, 700)
wlengths = np.arange(100, 700)
res_matrix = np.zeros((sfreqs.shape[0], wlengths.shape[0]))

for ii in range(len(sfreqs)):
    for jj in range(len(wlengths)):
        N = wlengths[jj]
        res_matrix[ii, jj] = sfreqs[ii] / N


#%% ------------------------------------------------------
#

plt.figure(figsize=(12, 9))
plt.subplots_adjust(hspace=0.7, top=0.925, bottom=0.075, left=0.05, right=0.95)

for ii in range(len(win_len)):
    ax = plt.subplot(3, len(win_len), ii+1)
    lemon_plotting.plot_sensor_spectrum(ax, win_len[ii][1], refraw, win_len[ii][0], base=1)
    ax.set_ylim(0, 2.2e-5)
    plt.text(0.9, 0.8, '{} Samples'.format(wls[ii]), transform=ax.transAxes, ha='right')
    if ii == 0:
        lemon_plotting.subpanel_label(ax, 'A - Segment Length', ha='left', yf=1.2)

    ax = plt.subplot(3, len(win_len), ii+6)
    lemon_plotting.plot_sensor_spectrum(ax, sample_rate[ii][1], refraw, sample_rate[ii][0], base=1)
    plt.text(0.9, 0.8, '{}Hz'.format(srs[ii]), transform=ax.transAxes, ha='right')
    ax.set_ylim(0, 8e-5)
    if ii == 0:
        lemon_plotting.subpanel_label(ax, 'B - Sample Rate', ha='left', yf=1.2)

    ax = plt.subplot(3, len(win_len), ii+11)
    lemon_plotting.plot_sensor_spectrum(ax, data_len[ii][1], refraw, data_len[ii][0], base=1)
    plt.text(0.9, 0.8, '{} Seconds'.format(dls[ii]), transform=ax.transAxes, ha='right')
    ax.set_ylim(0, 3e-5)
    if ii == 0:
        lemon_plotting.subpanel_label(ax, 'C - Data Length', ha='left', yf=1.2)

fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_periodogram-params.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_periodogram-params_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

plt.figure(figsize=(7, 6))
plt.pcolormesh(np.log(wlengths), sfreqs, res_matrix, cmap='gist_rainbow')
fx = np.linspace(100, 700, 7)
plt.xticks(np.log(fx), fx.astype(int))
plt.colorbar(label='Frequency Resolution (Hz)')
plt.ylabel('Sample Rate (Hz)')
plt.xlabel('Segment Length (Samples)')

fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_periodogram-resmatrix.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_periodogram-resmatrix_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)
