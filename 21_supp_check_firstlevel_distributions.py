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

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}/{subj}_preproc_raw.fif')
st = osl.utils.Study(fbase)

fname = st.get(subj='sub-010060')[0]

runname = fname.split('/')[-1].split('.')[0]
print('processing : {0}'.format(runname))

subj_id = osl.utils.find_run_id(fname)

raw = mne.io.read_raw_fif(fname, preload=True)

picks = mne.pick_types(raw.info, eeg=True, ref_meg=False)
chlabels = np.array(raw.info['ch_names'], dtype=h5py.special_dtype(vlen=str))[picks]

fbase = os.path.join(cfg['lemon_processed_data'], 'sub-010002', 'sub-010002_preproc_raw.fif')
reference = mne.io.read_raw_fif(fbase).pick_types(eeg=True)

#%% ----------------------------------------------------------
# GLM-Prep

# Make blink regressor
blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=None)

veog = np.abs(raw.get_data(picks='ICA-VEOG')[0, :])
heog = np.abs(raw.get_data(picks='ICA-HEOG')[0, :])

# Make task regressor
task = lemon_make_task_regressor({'raw': raw})

# Make bad-segments regressor
bads = lemon_make_bads_regressor(raw)
print('Bad segs in regressor {} / {}'.format(bads.sum(), len(bads)))

# Get data
XX = get_eeg_data(raw)
print(XX.shape)

# Run GLM-Periodogram

conds = {'Eyes Open': task > 0, 'Eyes Closed': task < 0}
covs = {'Linear Trend': np.linspace(0, 1, raw.n_times)}
confs = {'Bad Segs': bads,
            'V-EOG': veog, 'H-EOG': heog}
eo_val = np.round(np.sum(task == 1) / len(task), 3)
ec_val = np.round(np.sum(task == -1) / len(task), 3)
conts = [{'name': 'RestMean', 'values': {'Eyes Open': 0.5, 'Eyes Closed': 0.5}},
            {'name': 'Open>Closed', 'values': {'Eyes Open': 1, 'Eyes Closed': -1}}]

fs = raw.info['sfreq']

# Reduced model - no confounds or covariates
glmspec_mag = osl.glm.glm_spectrum(XX, fmin=1, fmax=95,
                            fs=fs,
                            fit_intercept=False,
                            nperseg=int(fs * 2),
                            mode='magnitude',
                            contrasts=conts,
                            reg_categorical=conds,
                            reg_ztrans=covs, reg_unitmax=confs,
                           standardise_data=False)
glmspec_mag = osl.glm.SensorGLMSpectrum(glmspec_mag, reference.info) # Store with standard channel info

# Reduced model - no confounds or covariates
glmspec_pow = osl.glm.glm_spectrum(XX, fmin=1, fmax=95,
                            fs=fs,
                            fit_intercept=False,
                            nperseg=int(fs * 2),
                            mode='psd',
                            contrasts=conts,
                            reg_categorical=conds,
                            reg_ztrans=covs, reg_unitmax=confs,
                            standardise_data=False)
glmspec_pow = osl.glm.SensorGLMSpectrum(glmspec_pow, reference.info) # Store with standard channel info

# Reduced model - no confounds or covariates
glmspec_log2 = osl.glm.glm_spectrum(XX, fmin=1, fmax=95,
                            fs=fs,
                            fit_intercept=False,
                            nperseg=int(fs * 2),
                            mode='log_psd',
                            contrasts=conts,
                            reg_categorical=conds,
                            reg_ztrans=covs, reg_unitmax=confs,
                            standardise_data=False)
glmspec_log2 = osl.glm.SensorGLMSpectrum(glmspec_log2, reference.info) # Store with standard channel info

# Reduced model - no confounds or covariates
glmspec_log = deepcopy(glmspec_pow)
glmspec_log.data.data = np.log(glmspec_log.data.data)
glmspec_log.model = glm.fit.OLSModel(glmspec_log.design, glmspec_log.data)


#%% ----------------------------------------------------------
# Figure

plt.figure(figsize=(16, 9))
plt.subplots_adjust(wspace=0.45, hspace=0.35, left=0.04, right=0.975)
ax = plt.subplot(2, 3, 1)
glmspec_pow.plot_joint_spectrum(0, base=0.5, freqs=(3, 9 , 25), ax=ax)
ax = plt.subplot(2, 3, 2)
glmspec_mag.plot_joint_spectrum(0, base=0.5, freqs=(3, 9 , 25), ax=ax, ylabel='Magnitude')
ax = plt.subplot(2, 3, 3)
glmspec_log.plot_joint_spectrum(0, base=0.5, freqs=(3, 9 , 25), ax=ax, ylabel='log(Power)')

ax = plt.subplot(2, 3, 4)
glmspec_pow.plot_joint_spectrum(1, base=0.5, freqs=(3, 9 , 25), ax=ax, metric='tstats')
ax = plt.subplot(2, 3, 5)
glmspec_mag.plot_joint_spectrum(1, base=0.5, freqs=(3, 9 , 25), ax=ax, metric='tstats')
ax = plt.subplot(2, 3, 6)
glmspec_log.plot_joint_spectrum(1, base=0.5, freqs=(3, 9 , 25), ax=ax, metric='tstats')


fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_first-level-distribution-comparison.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_first-level-distribution-comparison_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

eye

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
