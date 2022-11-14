"""This script produces a table summarising first level distributions."""

import os
from copy import deepcopy

import glmtools as glm
import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import sails

import lemon_plotting
from glm_config import cfg
from lemon_support import (get_eeg_data, lemon_make_bads_regressor,
                           lemon_make_blinks_regressor,
                           lemon_make_task_regressor)

#%% ----------------------------------------------------------
# GLM-Prep

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}_preproc_raw.fif')
st = osl.utils.Study(fbase)

fname = st.get(subj='sub-010060')[0]

runname = fname.split('/')[-1].split('.')[0]
print('processing : {0}'.format(runname))

subj_id = osl.utils.find_run_id(fname)

raw = mne.io.read_raw_fif(fname, preload=True)

picks = mne.pick_types(raw.info, eeg=True, ref_meg=False)
chlabels = np.array(raw.info['ch_names'], dtype=h5py.special_dtype(vlen=str))[picks]

#%% ----------------------------------------------------------
# GLM-Prep

# Make blink regressor
blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=None)

veog = raw.get_data(picks='ICA-VEOG')[0, :]**2
veog = veog > np.percentile(veog, 97.5)

heog = raw.get_data(picks='ICA-HEOG')[0, :]**2
heog = heog > np.percentile(heog, 97.5)

# Make task regressor
task = lemon_make_task_regressor({'raw': raw})

# Make bad-segments regressor
bads_raw = lemon_make_bads_regressor(raw, mode='raw')
bads_diff = lemon_make_bads_regressor(raw, mode='diff')

# Get data
XX = get_eeg_data(raw).T
print(XX.shape)

# Run GLM-Periodogram
conds = {'Eyes Open': task == 1, 'Eyes Closed': task == -1}
covs = {'Linear Trend': np.linspace(0, 1, raw.n_times)}
confs = {'Bad Segments': bads_raw,
         'Bad Segments Diff': bads_diff,
         'V-EOG': veog, 'H-EOG': heog}
conts = [{'name': 'Mean', 'values': {'Eyes Open': 0.5, 'Eyes Closed': 0.5}},
         {'name': 'Open < Closed', 'values': {'Eyes Open': 1, 'Eyes Closed': -1}}]

fs = raw.info['sfreq']

#%% ----------------------------------------------------------
# GLM-Run


# Simple model
f, c, v, extras = sails.stft.glm_periodogram(XX, axis=0,
                                             fit_constant=False,
                                             conditions=conds,
                                             covariates=covs,
                                             confounds=confs,
                                             contrasts=conts,
                                             nperseg=int(fs*2),
                                             noverlap=int(fs),
                                             fmin=0.1, fmax=100,
                                             fs=fs, mode='magnitude',
                                             fit_method='glmtools')
model, design, data = extras

data_pow = deepcopy(data)
data_pow.data = data_pow.data**2
model_pow = glm.fit.OLSModel(design, data_pow)

model_mag = glm.fit.OLSModel(design, data)

data_logpow = deepcopy(data)
data_logpow.data = np.log(data_logpow.data**2)
model_logpow = glm.fit.OLSModel(design, data_logpow)

#%% ----------------------------------------------------------
# Figure

ff = 19
chan = 24


def quick_decorate(ax):
    """Decorate an axes."""
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_ylabel('Num Segments')


plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(3, 3, 1)
plt.plot(data_pow.data.mean(axis=0))
lemon_plotting.decorate_spectrum(plt.gca(), ylabel='Power')
plt.title('Data Spectrum')
lemon_plotting.subpanel_label(plt.gca(), 'A')
plt.subplot(3, 3, 2)
plt.hist(data_pow.data[:, ff, chan], 64)
quick_decorate(plt.gca())
plt.xlabel('Power')
plt.title('Data Distribution\n(single channel and freq)')
plt.subplot(3, 3, 3)
resids = model_pow.get_residuals(data_pow.data)
plt.hist(resids[:, ff, chan], 64)
quick_decorate(plt.gca())
plt.xlabel('Power')
plt.title('Residual Distribution\n(single channel and freq)')

plt.subplot(3, 3, 4)
plt.plot(data.data.mean(axis=0))
lemon_plotting.decorate_spectrum(plt.gca(), ylabel='Magnitude')
lemon_plotting.subpanel_label(plt.gca(), 'B')
plt.subplot(3, 3, 5)
plt.hist(data.data[:, ff, chan], 64)
quick_decorate(plt.gca())
plt.xlabel('Magnitude')
plt.subplot(3, 3, 6)
resids = model_mag.get_residuals(data.data)
plt.hist(resids[:, ff, chan], 64)
quick_decorate(plt.gca())
plt.xlabel('Magnitude')

plt.subplot(3, 3, 7)
plt.plot(data_logpow.data.mean(axis=0))
lemon_plotting.decorate_spectrum(plt.gca(), ylabel='log(Power)')
lemon_plotting.subpanel_label(plt.gca(), 'C')
plt.subplot(3, 3, 8)
plt.hist(data_logpow.data[:, ff, chan], 64)
quick_decorate(plt.gca())
plt.xlabel('log(Power)')
plt.subplot(3, 3, 9)
resids = model_logpow.get_residuals(data_logpow.data)
plt.hist(resids[:, ff, chan], 64)
quick_decorate(plt.gca())
plt.xlabel('log(Power)')

plt.savefig('dist_check.png')

fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_first-level-distribution-check.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_first-level-distribution-check_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)
