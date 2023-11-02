"""This script creates figures 1."""

import os
from copy import deepcopy

import dill
import glmtools as glm
import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import sails
from anamnesis import obj_from_hdf5file

import lemon_plotting
from glm_config import cfg
from lemon_support import (get_eeg_data, plot_design, lemon_make_task_regressor, 
                           lemon_make_bads_regressor, lemon_set_standard_montage)

outdir = cfg['lemon_figures']
subj_id = 'sub-010060'

#%% --------------------------------------------------
# Load dataset

fbase = os.path.join(cfg['lemon_processed_data'], 'sub-010002', 'sub-010002_preproc_raw.fif')
reference = mne.io.read_raw_fif(fbase).pick_types(eeg=True)

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}/{subj}_preproc_raw.fif')
st_raw = osl.utils.Study(fbase)

raw = mne.io.read_raw_fif(st_raw.get(subj=subj_id)[0], preload=True)
raw.crop(tmin=29)

sensor = 'Pz'
ch_ind = raw.info['ch_names'].index(sensor)


# -------------------
veog = np.abs(raw.get_data(picks='VEOG')[0, :])#**2
heog = np.abs(raw.get_data(picks='HEOG')[0, :])#**2
# Make task regressor
task = lemon_make_task_regressor({'raw': raw})
# Make bad-segments regressor
#bads_raw = lemon_make_bads_regressor(raw, mode='raw')
#bads_diff = lemon_make_bads_regressor(raw, mode='diff')
#bads = np.logical_or(bads_raw, bads_diff)
bads = lemon_make_bads_regressor(raw)



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
XX = get_eeg_data(raw)
glmsp = osl.glm.glm_spectrum(XX, fmin=1, fmax=95,
                            fs=fs,
                            fit_intercept=False,
                            nperseg=int(fs * 2),
                            mode='magnitude',
                            contrasts=conts,
                            reg_categorical=conds,
                            reg_ztrans=covs, reg_unitmax=confs,
                            standardise_data=True)
glmsp = osl.glm.SensorGLMSpectrum(glmsp, reference.info) # Store with standard channel info
glmsp = lemon_set_standard_montage(glmsp)
# Load GLM Results
#st = osl.utils.Study(os.path.join(cfg['lemon_glm_data'],'{subj_id}_preproc_raw-glm-spectrum_model-{model}_data.pkl'))
#fname = st.get(subj=subj_id)[0]

#glmsp = osl.glm.read_glm_spectrum(fname)

# This shouldn't be necessary...
#model.design_matrix = design.design_matrix
#model.regressor_list = design.regressor_list

fout = os.path.join(outdir, '{subj_id}_firstlevel-design.png'.format(subj_id=subj_id))
fig = glmsp.design.plot_summary(figargs={'figsize': (16, 12)}, show=False)
fig.savefig(fout, dpi=100, transparent=True)

fout = os.path.join(outdir, '{subj_id}_firstlevel-efficiency.png'.format(subj_id=subj_id))
fig = glmsp.design.plot_efficiency()
fig.savefig(fout, dpi=100, transparent=True)

#%% ---------------------------------------------------------
# Single channel example figure 1

sensor = 'Pz'
fs = glmsp.info['sfreq']

ch_ind = glmsp.info['ch_names'].index(sensor)
YY = XX[ch_ind, :]

# Extract data segment
inds = np.arange(140*fs, 160*fs)
inds = np.arange(770*fs, 790*fs).astype(int)
inds = np.arange(520*fs, 550*fs).astype(int) - int(28*fs)

#eog = dataset['raw'].copy().pick_types(eog=True, eeg=False)
#eog.filter(l_freq=1, h_freq=25, picks='eog')
#eog = eog.get_data(picks='eog')[:, inds].T
time = np.linspace(0, YY.shape[0]/fs, YY.shape[0])

config = deepcopy(glmsp.config)
config_flat = deepcopy(glmsp.config)
config_flat.window = None
config_flat.detrend_func = None


# Apply sliding window to subset indices to get corresponding stft windows
subset = np.zeros((YY.shape[0],))
subset[inds] = 1
xsubset = sails.stft.apply_sliding_window(subset, **config_flat.sliding_window_args)
stft_inds = np.where(xsubset.sum(axis=1) > 200)[0]

stft = glmsp.data.data[stft_inds, ch_ind, :]
stft_time = np.arange(config.nperseg/2, YY.shape[0] - config.nperseg/2 + 1,
                      config.nperseg - config.noverlap) / float(fs)
stft_time = stft_time[stft_inds] - stft_time[stft_inds[0]] + 1

stft_dm = glmsp.design.design_matrix[stft_inds, :]

# Exaggerate linear trend to help visualisation
stft_dm[:, 2] = np.linspace(-1, 1, len(stft_inds))

f = glmsp.f #sails.stft._set_freqvalues(config.nfft, config.fs, 'onesided')
#fidx = (f >= config.fmin) & (f <= config.fmax)
#f = f[fidx]

inds = np.arange(stft_inds[0]*250 - 125, stft_inds[-1]*250 + 250)

# prep sqrt(f) axes
fx, ftl, ft = lemon_plotting.prep_scaled_freq(0.5, f)

#%% ------------------------------------------------------------
# Make figure 2

wlagX = sails.stft.apply_sliding_window(YY, **config.sliding_window_args)[stft_inds, :]
lagX = sails.stft.apply_sliding_window(YY, **config_flat.sliding_window_args)[stft_inds, :]

panel_label_height = 1.075
plt.figure(figsize=(16, 9))

ax_ts = plt.axes([0.05, 0.1, 0.4, 0.8])
ax_tf = plt.axes([0.45, 0.1, 0.16, 0.8])
ax_tf_cb = plt.axes([0.47, 0.065, 0.12, 0.01])
ax_des = plt.axes([0.7, 0.1, 0.25, 0.8])
ax_des_cb = plt.axes([0.96, 0.15, 0.01, 0.2])

# Plot continuous time series
scale = 1 / (np.std(YY) * 4)
ax_ts.plot(scale*YY[inds], time[inds] - time[inds[0]], 'k', lw=0.5)

# Plot window markers + guidelines
for ii in range(stft.shape[0]):
    jit = np.remainder(ii, 3) / 5
    ax_ts.plot((2+jit, 2+jit), (ii, ii+2), lw=4, solid_capstyle="butt")
    ax_ts.plot((0, 14), (ii+1, ii+1), lw=0.5, color=[0.8, 0.8, 0.8])

ax_ts.set_prop_cycle(None)
x = np.linspace(4, 8, 500)
ax_ts.plot(x, scale*lagX.T + np.arange(stft.shape[0])[None, :] + 1, lw=0.8)

ax_ts.plot(x, np.zeros_like(x), 'k')
ax_ts.text(x[0], -0.1, '2 Seconds', ha='left', va='top')

ax_ts.set_prop_cycle(None)
x = np.linspace(9, 13, 500)
ax_ts.plot(x, scale*wlagX.T + np.arange(stft.shape[0])[None, :] + 1, lw=0.8)
ax_ts.set_ylim(0, stft.shape[0]+1)
ax_ts.set_xlim(-2, 14)

ax_ts.plot(x, np.zeros_like(x), 'k')
ax_ts.text(x[0], -0.1, '2 Seconds', ha='left', va='top')

for tag in ['top', 'right', 'bottom']:
    ax_ts.spines[tag].set_visible(False)
ax_ts.set_xticks([])
ax_ts.set_yticks(np.linspace(0, stft.shape[0]+1, 9))
ax_ts.set_ylabel('Time (seconds)')

lemon_plotting.subpanel_label(ax_ts, 'A', xf=-0.02, yf=panel_label_height)
ax_ts.text(0.1, panel_label_height, '\nRaw EEG\nChannel: {}'.format(sensor),
           ha='center', transform=ax_ts.transAxes, fontsize='large')
lemon_plotting.subpanel_label(ax_ts, 'B', xf=0.25, yf=panel_label_height)
ax_ts.text(0.4, panel_label_height, 'Segmented EEG',
           ha='center', transform=ax_ts.transAxes, fontsize='large')
lemon_plotting.subpanel_label(ax_ts, 'C', xf=0.7, yf=panel_label_height)
ax_ts.text(0.825, panel_label_height, "'Windowed' EEG",
           ha='center', transform=ax_ts.transAxes, fontsize='large')

pcm = ax_tf.pcolormesh(fx, stft_time, stft, cmap='magma_r')
ax_tf.set_xticks(ft)
ax_tf.set_xticklabels(ftl)
plt.colorbar(pcm, cax=ax_tf_cb, orientation='horizontal')
ax_tf_cb.set_title('Magnitude')
for tag in ['bottom', 'right']:
    ax_tf.spines[tag].set_visible(False)
ax_tf.xaxis.tick_top()
ax_tf.set_xlabel('Frequency (Hz)')
ax_tf.xaxis.set_label_position('top')
ax_tf.set_yticks(np.linspace(0, stft.shape[0]+1, 7))
ax_tf.set_yticklabels([])
ax_tf.text(0.475, panel_label_height, 'Short Time Fourier Tranform',
           va='center', ha='center', transform=ax_tf.transAxes, fontsize='large')
lemon_plotting.subpanel_label(ax_tf, 'D', xf=-0.05, yf=panel_label_height)

pcm = plot_design(ax_des, stft_dm, glmsp.design.regressor_names)
for ii in range(len(glmsp.design.regressor_names)):
    ax_des.text(0.5+ii, stft.shape[0], glmsp.design.regressor_names[ii],
                ha='left', va='bottom', rotation=20)
ax_des.set_yticks(np.linspace(0, stft.shape[0]+1, 9)-0.5,
                  np.linspace(0, stft.shape[0]+1, 9).astype(int))
ax_des.text(0.25, panel_label_height, 'GLM Design Matrix',
            ha='center', transform=ax_des.transAxes, fontsize='large')
plt.colorbar(pcm, cax=ax_des_cb)
lemon_plotting.subpanel_label(ax_des, 'E', xf=-0.02, yf=panel_label_height)


fout = os.path.join(outdir, '{subj_id}_single-channel_glm-top.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=300, transparent=True)
fout = os.path.join(outdir, '{subj_id}_single-channel_glm-top_low-res.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=100, transparent=True)

#%% ---------------------------------------------
# Bottom of figure 1

tstat_args = {'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}
tstat_args_for_plotting = {'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 0}
tstat_args = {}
tstat_args_for_plotting = {}

con_ind = [1, 7]

fig = plt.figure(figsize=(16, 6))
ax = plt.axes([0.075, 0.2, 0.25, 0.6])
ax.plot(fx, glmsp.model.copes[2, ch_ind, :], label='Eyes Open')
ax.plot(fx, glmsp.model.copes[3, ch_ind, :], label='Eyes Closed')
ax.set_xticks(ft)
ax.set_xticklabels(ftl)
ax.set_ylabel('Magnitude')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylim(0)
lemon_plotting.subpanel_label(ax, 'F', xf=-0.02, yf=1.1)
plt.legend(frameon=False)
ax.text(0.5, 1.1, 'GLM cope-spectrum', ha='center', transform=ax.transAxes, fontsize='large')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)

for ii in range(len(con_ind)):
    if ii > 0:
        continue
    plt.axes([0.4+ii*0.3, 0.55, 0.15, 0.25])
    plt.plot(fx, glmsp.model.copes[con_ind[ii], ch_ind, :])
    lemon_plotting.subpanel_label(plt.gca(), chr(71+ii), xf=-0.02, yf=1.3)
    plt.xticks(ft[::2], ftl[::2])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.title('cope-spectrum\n')
    plt.axes([0.4+ii*0.3, 0.15, 0.15, 0.25])
    plt.plot(fx, glmsp.model.varcopes[con_ind[ii], ch_ind, :])
    plt.xticks(ft[::2], ftl[::2])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.title('varcope-spectrum\n')
    plt.xlabel('Frequency (Hz)')

    #ax = plt.axes([0.4+ii*0.3, 0.1, 0.25, 0.5])
    ax = plt.axes([0.65, 0.2, 0.25, 0.6])

    ts = glm.fit.get_tstats(glmsp.model.copes[con_ind[ii], ch_ind, :],
                            glmsp.model.varcopes[con_ind[ii], ch_ind, :],
                            **tstat_args_for_plotting)
    ax.plot(fx, ts)
    lemon_plotting.subpanel_label(ax, chr(72+ii), xf=-0.02, yf=1.1)
    name = glmsp.model.contrast_names[con_ind[ii]]
    ax.text(0.5, 1.7, f'Contrast : {name}', ha='center', transform=ax.transAxes, fontsize='large')
    ax.set_xticks(ft)
    ax.set_xticklabels(ftl)
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.title('t-spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('t statistic')

fout = os.path.join(outdir, '{subj_id}_no-smo_single-channel_glm-bottom.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=300, transparent=True)
fout = os.path.join(outdir, '{subj_id}_no-smo_single-channel_glm-bottom_low-res.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=100, transparent=True)

#%% --------------------------------------------------------
# Whole head GLM single subject figure



#tstat_args = {'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}
tstat_args = {}
#ts = model.get_tstats(**tstat_args)


ll = [['Rec Start', 'Rec End'],
      ['Good Seg', 'Bad Seg'],
      ['Low V-EOG Activity', 'High V-EOG Activity'],
      ['Low H-EOG Activity', 'High H-EOG Activity']]

col_heads = ['Mean', 'Linear Trend', 'Rest Condition', 'Bad Segments', 'VEOG', 'HEOG']

plt.figure(figsize=(16, 16))
ax = plt.axes([0.075, 0.6, 0.175, 0.3])
glmsp.plot_joint_spectrum(2, ax=ax, base=0.5, ylabel='Magnitude', freqs=(9, 22))
lemon_plotting.subpanel_label(ax, chr(65), yf=1.1)

ax = plt.axes([0.3125, 0.6, 0.175, 0.3])
glmsp.plot_joint_spectrum(3, ax=ax, base=0.5, ylabel='Magnitude', freqs=(9, 22))
lemon_plotting.subpanel_label(ax, chr(66), yf=1.1)

# Plot Open > Closd
ax = plt.axes([0.55, 0.6, 0.175, 0.3])
glmsp.plot_joint_spectrum(1, ax=ax, base=0.5, ylabel='t-values', metric='tstats', freqs=(9, 22))
lemon_plotting.subpanel_label(ax, chr(67), yf=1.1)

ax = plt.axes([0.775, 0.6, 0.2, 0.2])
lemon_plotting.plot_channel_layout2(ax, glmsp.info, size=100)

# Covariate freqs of interest
fois = [[9, 23], # linear
        [2, 64], # bad segs
        [2, 9, ], # VEOG
        [9, 64], # HEOG
        ]

# Plot covariates
for ii in range(4):
    ax = plt.axes([0.07+ii*0.24, 0.25, 0.15, 0.25])

    #lemon_plotting.plot_joint_spectrum(ax, glmsp.model.tstats[ii+8, :, :], rawref, xvect=f,
    #                                     freqs=fois[ii], base=0.5, topo_scale=None,
    #                                     ylabel='pseudo t-statistic', title=model.contrast_names[ii+8])
    glmsp.plot_joint_spectrum(ii+4, ax=ax, base=0.5, ylabel='t-values', metric='tstats', freqs=fois[ii])
    lemon_plotting.subpanel_label(ax, chr(68+ii), yf=1.1)

    ax2 = plt.axes([0.07+ii*0.24, 0.07, 0.15, 0.2*2/3])
    #ax2.set_ylim(0, 1.5e-5)
    proj, llabels = glmsp.model.project_range(ii+2, nsteps=2)
    ax2.plot(fx, proj.mean(axis=1).T, lw=2)
    ax2.set_xticks(ft)
    ax2.set_xticklabels(ftl)
    ylabel = 'Magnitude' if ii == 0 else ''
    lemon_plotting.decorate_spectrum(ax2, ylabel=ylabel)
    ax2.legend(ll[ii], frameon=False, fontsize=8)

fout = os.path.join(outdir, '{subj_id}_no-smo_whole-head-glm-summary.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=300, transparent=True)
fout = os.path.join(outdir, '{subj_id}_no-smo_whole-head-glm-summary_low-res.png'.format(subj_id=subj_id))
plt.savefig(fout, dpi=100, transparent=True)
