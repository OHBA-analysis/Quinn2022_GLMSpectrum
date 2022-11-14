"""This script creates a summary of the residual autocorrelation."""

import os

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


# FROM STATSMODELS
def durbin_watson(resids, axis=0):
    r"""
    Calculate the Durbin-Watson statistic.

    Parameters
    ----------
    resids : array_like
        Data for which to compute the Durbin-Watson statistic. Usually
        regression model residuals.
    axis : int, optional
        Axis to use if data has more than 1 dimension. Default is 0.

    Returns
    -------
    dw : float, array_like
        The Durbin-Watson statistic.

    Notes
    -----
    The null hypothesis of the test is that there is no serial correlation
    in the residuals.
    The Durbin-Watson test statistic is defined as:

    .. math::

       \sum_{t=2}^T((e_t - e_{t-1})^2)/\sum_{t=1}^Te_t^2

    The test statistic is approximately equal to 2*(1-r) where ``r`` is the
    sample autocorrelation of the residuals. Thus, for r == 0, indicating no
    serial correlation, the test statistic equals 2. This statistic will
    always be between 0 and 4. The closer to 0 the statistic, the more
    evidence for positive serial correlation. The closer to 4, the more
    evidence for negative serial correlation.

    """
    # This function has been borrowed from statsmodels
    resids = np.asarray(resids)
    diff_resids = np.diff(resids, 1, axis=axis)
    dw = np.sum(diff_resids**2, axis=axis) / np.sum(resids**2, axis=axis)
    return dw


#%% ----------------------------------------------------------
# GLM-Prep

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}_preproc_raw.fif')
st = osl.utils.Study(fbase)

fname = st.get(subj='sub-010060')[0]

runname = fname.split('/')[-1].split('.')[0]
print('processing : {0}'.format(runname))

subj_id = osl.utils.find_run_id(fname)

raw = mne.io.read_raw_fif(fname, preload=True)

icaname = fname.replace('preproc_raw.fif', 'ica.fif')
ica = mne.preprocessing.read_ica(icaname)

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

npersegs = np.array([100, fs, fs*2, fs*5])

xc1 = []
dw1 = []
ff1 = []

xc2 = []
dw2 = []
ff2 = []

xc3 = []
dw3 = []
ff3 = []

for nn in range(len(npersegs)):
    # Full model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    fit_constant=False,
                                                                    conditions=conds,
                                                                    covariates=covs,
                                                                    confounds=confs,
                                                                    contrasts=conts,
                                                                    nperseg=int(npersegs[nn]),
                                                                    noverlap=1,
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')

    resids = extras[0].get_residuals(extras[2].data)

    xc_iter = np.zeros((resids.shape[1], 61))
    dw_iter = np.zeros((resids.shape[1], 61))

    for ii in range(resids.shape[1]):
        for jj in range(61):
            xc_iter[ii, jj] = np.corrcoef(resids[1:, ii, jj], resids[:-1, ii, jj])[1, 0]
            dw_iter[ii, jj] = durbin_watson(resids[:, ii, jj])

    dw1.append(dw_iter)
    xc1.append(xc_iter)
    ff1.append(freq_vect)

    # Full model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    fit_constant=False,
                                                                    conditions=conds,
                                                                    covariates=covs,
                                                                    confounds=confs,
                                                                    contrasts=conts,
                                                                    nperseg=int(npersegs[nn]),
                                                                    noverlap=int(npersegs[nn]//2),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')

    resids = extras[0].get_residuals(extras[2].data)

    xc_iter = np.zeros((resids.shape[1], 61))
    dw_iter = np.zeros((resids.shape[1], 61))

    for ii in range(resids.shape[1]):
        for jj in range(61):
            xc_iter[ii, jj] = np.corrcoef(resids[1:, ii, jj], resids[:-1, ii, jj])[1, 0]
            dw_iter[ii, jj] = durbin_watson(resids[:, ii, jj])

    dw2.append(dw_iter)
    xc2.append(xc_iter)
    ff2.append(freq_vect)

    # Full model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(XX, axis=0,
                                                                    fit_constant=False,
                                                                    conditions=conds,
                                                                    covariates=covs,
                                                                    confounds=confs,
                                                                    contrasts=conts,
                                                                    nperseg=int(npersegs[nn]),
                                                                    noverlap=3*int(npersegs[nn]//4),
                                                                    fmin=0.1, fmax=100,
                                                                    fs=fs, mode='magnitude',
                                                                    fit_method='glmtools')

    resids = extras[0].get_residuals(extras[2].data)[10:, :, :]

    xc_iter = np.zeros((resids.shape[1], 61))
    dw_iter = np.zeros((resids.shape[1], 61))

    for ii in range(resids.shape[1]):
        for jj in range(61):
            xc_iter[ii, jj] = np.corrcoef(resids[1:, ii, jj], resids[:-1, ii, jj])[1, 0]
            dw_iter[ii, jj] = durbin_watson(resids[:, ii, jj])

    dw3.append(dw_iter)
    xc3.append(xc_iter)
    ff3.append(freq_vect)

#%% --------------------------------------------------------------

plt.figure(figsize=(16, 16))

for ii in range(len(npersegs)):
    plt.subplot(3, len(npersegs), ii+1)
    plt.pcolormesh(np.arange(61), ff1[ii], np.abs(dw1[ii]), vmin=0, vmax=4, cmap='RdBu')
    if ii == len(npersegs) - 1:
        plt.colorbar()
    plt.title('nperseg: {0}, nstep: {1}'.format(int(npersegs[ii]), int(npersegs[ii]-1)))

    if ii == 0:
        plt.ylabel('Frequency (Hz)')
        lemon_plotting.subpanel_label(plt.gca(), 'A')

    plt.subplot(3, len(npersegs), ii+len(npersegs)+1)
    plt.pcolormesh(np.arange(61), ff2[ii], np.abs(dw2[ii]), vmin=0, vmax=4, cmap='RdBu')
    if ii == len(npersegs) - 1:
        plt.colorbar()
    plt.title('nperseg: {0}, nstep: {1}'.format(int(npersegs[ii]), int(npersegs[ii]//2)))

    if ii == 0:
        plt.ylabel('Frequency (Hz)')
        lemon_plotting.subpanel_label(plt.gca(), 'B')

    plt.subplot(3, len(npersegs), ii+2*len(npersegs)+1)
    plt.pcolormesh(np.arange(61), ff3[ii], np.abs(dw3[ii]), vmin=0, vmax=4, cmap='RdBu')
    if ii == len(npersegs) - 1:
        plt.colorbar()
    plt.xlabel('Sensor')
    plt.title('nperseg: {0}, nstep: {1}'.format(int(npersegs[ii]), int(npersegs[ii]//4)))

    if ii == 0:
        plt.ylabel('Frequency (Hz)')
        lemon_plotting.subpanel_label(plt.gca(), 'C')

fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_first-level-residualautocorrelation.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_first-level-residualautocorrelation_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)
