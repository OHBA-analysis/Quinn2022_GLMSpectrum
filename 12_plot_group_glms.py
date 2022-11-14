"""This script creates the group level figures."""

import os
from copy import deepcopy

import dill
import glmtools as glm
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
from anamnesis import obj_from_hdf5file

import lemon_plotting
from glm_config import cfg

#%% ---------------------------------------------------
# Load single subject for reference

fbase = os.path.join(cfg['lemon_processed_data'], 'sub-010002_preproc_raw.fif')
raw = mne.io.read_raw_fif(fbase).pick_types(eeg=True)

inputs = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata.hdf5')
inputs_reduced = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata_reduced.hdf5')

data = obj_from_hdf5file(inputs, 'data')
reduced = obj_from_hdf5file(inputs_reduced, 'data')

st = osl.utils.Study(os.path.join(cfg['lemon_glm_data'], '{subj}_preproc_raw_glm-data.hdf5'))
freq_vect = h5py.File(st.match_files[0], 'r')['freq_vect'][()]

glm_fname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-data.hdf5')
gmodel = obj_from_hdf5file(glm_fname, 'model')
design = obj_from_hdf5file(glm_fname, 'design')
clean_data = obj_from_hdf5file(glm_fname, 'data')

#%% ------------------------------------------------------
# R-squared distributions across participants and regressors

r2_path = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_rsquared.npy')
r2 = np.load(r2_path)

fl_regressor_names = ['Constant', 'Eyes Open', 'Eyes Closed',
                     'Linear Trend', 'Bad Segs', 'Bad Segs Diff', 'V-EOG', 'H-EOG']

plt.figure(figsize=(16, 11))
plt.subplots_adjust(wspace=0.4, hspace=0.5, top=0.95, bottom=0.05)
labels = []

group_model_selection = glm.fit.run_regressor_selection(design, clean_data)

# Probably a simpler way to do this...
subpanels = (np.arange(15)+1).reshape(3,5)[:, :3].reshape(-1)

fx, xticklabels, xticks = lemon_plotting.prep_scaled_freq(0.5, freq_vect)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=[0.6, 0.6, 0.6], lw=0.25),
                Line2D([0], [0], color='k', lw=1)]

# Plot first level regressors across subject
for ii in range(9):
    ax = plt.subplot(3, 5, subpanels[ii])

    ax.plot(fx, r2[:, ii, :, :].mean(axis=2).T, color=[0.6, 0.6, 0.6], lw=0.25)
    ax.plot(fx, r2[:, ii, :, :].mean(axis=(0,2)), 'k')
    ax.set_ylim(0, 90)
    lemon_plotting.decorate_spectrum(ax, ylabel='R-Squared (%)')
    ax.set_xticks(xticks[::2], xticklabels[::2])

    if ii == 0:
        lemon_plotting.subpanel_label(ax, chr(65+ii))
    elif ii == 4:
        ax.legend(custom_lines, ['Single Dataset', 'Group Mean'], frameon=False)

    title = 'Full Model' if ii == 0 else fl_regressor_names[ii-1] + ' Only'
    ax.set_title(title)

# Plot an overall distribution
ax = plt.axes([0.65, 0.2, 0.3, 0.6])
for ii in range(9):
    x = r2[:, ii, :, :].mean(axis=(1,2))
    y = np.random.normal(ii+1, 0.05, size=len(x))
    plt.plot(y, x, 'r.', alpha=0.2)

boxdata = [r2[:, ii, :, :].mean(axis=(1,2)) for ii in range(9)]
h = plt.boxplot(boxdata, vert=True, showfliers=False)
plt.xticks(np.arange(1, 10), ['Full Model'] + fl_regressor_names, rotation=45)
for tag in ['top', 'right']:
    ax.spines[tag].set_visible(False)
ax.set_ylabel('R-Squared (%)')
ax.set_xlabel('Model')
ax.set_ylim(0)
lemon_plotting.subpanel_label(ax, chr(66))

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_rsquare-firstlevels.png')
plt.savefig(fout, transparent=True, dpi=300)   
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_rsquare-firstlevels_low-res.png')
plt.savefig(fout, transparent=True, dpi=100) 

#%% ------------------------------------------------------
# Group-level R-squared values across channels and regressors

fois = [(4,35),
        (4,35),
        (4,35),
        (16,25),
        (16,35),
        (10,35)]

plt.figure(figsize=(16,6))
for ii in range(6):
    ax = plt.subplot(2,6,ii+7)
    yl = (0, 100) if ii < 3 else (0, 7.5)
    ax.set_ylim(*yl)
    ax.set_ylim(0)

    change = group_model_selection[ii].r_square[0, 0, :, :] * 100
    ylabel = 'R-Squared (%)' if ii == 0 else ','
    lemon_plotting.plot_joint_spectrum(ax, change, raw, freq_vect, ylabel=ylabel,
                                       freqs=fois[ii], base=0.5, topo_scale=None, ylim=yl,
                                       xtick_skip=2) 
    title = 'Full Model' if ii == 0 else gmodel.regressor_names[ii-1] + ' Only'
    ax.set_title(title)
    lemon_plotting.subpanel_label(ax, chr(65+ii))

ax = plt.axes([0.825, 0.4, 0.2, 0.2])
lemon_plotting.plot_channel_layout(ax, raw)


fout = os.path.join(cfg['lemon_figures'], 'lemon-group_rsquare-group.png')
plt.savefig(fout, transparent=True, dpi=300)    
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_rsquare-group_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)  

#%% ------------------------------------------------------
# Effect of covariates across first-levels

sw = np.array(data.info['shapiro'])
sw_reduced = np.array(reduced.info['shapiro'])
r2 = np.array(data.info['r2'])
r2_reduced = np.array(reduced.info['r2'])

labs = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
fontargs = {'fontsize': 'large', 'fontweight': 'bold', 'color': 'r', 'ha': 'center', 'va': 'center'}

plt.figure(figsize=(16, 9))

ppts = [43, 145, 68, 62, 112]
plt.axes((0.1, 0.5, 0.4*(9/16), 0.4))
plt.plot(r2_reduced, r2, 'ko')
plt.plot(r2_reduced[ppts], r2[ppts], 'ro')
plt.text(r2_reduced[ppts][0]-0.01, r2[ppts][0]+0.02, labs[0], **fontargs)
plt.text(r2_reduced[ppts][1]-0.02, r2[ppts][1]+0.02, labs[1], **fontargs)
plt.text(r2_reduced[ppts][2]+0.02, r2[ppts][2]-0.035, labs[2],      **fontargs)
plt.text(r2_reduced[ppts][3]+0.035, r2[ppts][3], labs[3],      **fontargs)
plt.text(r2_reduced[ppts][4]-0.02, r2[ppts][4]+0.02, labs[4], **fontargs)

plt.plot((0, 1), (0, 1), 'k')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('reduced Model')
plt.ylabel('Full Model')
plt.xlim(0.2, 0.8)
plt.ylim(0.2, 0.8)
lemon_plotting.subpanel_label(plt.gca(), 'A')
plt.title('R-Squared\nhigh values indicate greater variance explained')

plt.axes((0.375, 0.5, 0.05, 0.4))
plt.boxplot(r2-r2_reduced)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylim(-0.5, 0.5)
plt.xticks([1], ['Full Model minus \nreduced Model'])

plt.axes((0.6, 0.5, 0.4*(9/16), 0.4))
plt.hist(np.array(reduced.info['aic']) - np.array(data.info['aic']), np.linspace(-500, 500, 64))
plt.xlabel('Reduced AIC - Full AIC')
plt.ylabel('Num Datasets')

for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)

lemon_plotting.subpanel_label(plt.gca(), 'B')
plt.title("Akaike's Information Criterion\nmore positive values indicate a better 'full' model")


fx = lemon_plotting.prep_scaled_freq(0.5, freq_vect,)
labs = ['C i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
for ii in range(len(ppts[:5])):
    ax = plt.axes((0.15+ii*0.16, 0.07, 0.1, 0.3))
    ax.plot(fx[0], reduced.data[ppts[ii], 0, :, :].mean(axis=1))
    ax.plot(fx[0], data.data[ppts[ii], 0, :, :].mean(axis=1))
    ax.set_xticks(fx[2][::2])
    ax.set_xticklabels(fx[1][::2])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.ylim(0, 1.2e-5)
    lemon_plotting.subpanel_label(plt.gca(), labs[ii])
    if ii == 0:
        plt.ylabel('Magnitude')
    plt.xlabel('Frequency (Hz)')
plt.legend(['reduced Model', 'Full Model'], frameon=False)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_stats-summary.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_stats-summary_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

#%% ---------------------------------------------------------------------
# Plot main group results
fl_contrast_names = ['OverallMean', 'RestMean',
                     'Eyes Open AbsEffect', 'Eyes Closed AbsEffect', 'Open > Closed',
                     'Constant', 'Eyes Open', 'Eyes Closed',
                     'Linear Trend', 'Bad Segs', 'Bad Segs Diff', 'V-EOG', 'H-EOG']

tstat_args = {'varcope_smoothing': 'medfilt',
              'window_size': 15, 'smooth_dims': 1}


# Each line is  (group cont, firstlevel cont, group regressor, firstlevel regressor, savename)
to_permute = (
    (0, 4, (0, 1), 4, 'Mean_OpenClosed'),
    (1, 0, (0, 1), (0, 1, 2), 'YoungOld_Mean'),
    (1, 4, (0, 1), 4, 'YoungOld_OpenClosed'),
    (0, 8, (0, 1), 3, 'Mean_Linear'),
    (0, 9, (0, 1), 4, 'Mean_BadSeg'),
    (0, 10, (0, 1), 5, 'Mean_DiffBadSeg'),
    (0, 11, (0, 1), 6, 'Mean_VEOG'),
    (0, 12, (0, 1), 7, 'Mean_HEOG'),
    (4, 0, 2, (0, 1, 2), 'Sex_Mean'),
    (5, 0, 3, (0, 1, 2), 'BrainVol_Mean'),
    (6, 0, 4, (0, 1, 2), 'GreyVol_Mean'),
)


def plot_perm_column(clean_data, gmodel, name, ax1, ax2, title=None, subpanel=None):
    """Plot a two panel column showing a t-spectrum + stats."""
    icon = np.where([pp[4] == name for pp in to_permute])[0][0]
    gl_con, fl_con, gl_reg, fl_reg, p_name = to_permute[icon]
    print(to_permute[icon])

    fl_mean_data = deepcopy(clean_data)
    fl_mean_data.data = clean_data.data[:, fl_con, :, :]  # first level contrast of interest

    dill_file = os.path.join(cfg['lemon_glm_data'], 'lemon-group_perms-{0}.pkl'.format(p_name))
    print('Loading {}'.format(dill_file))
    P = dill.load(open(dill_file, 'rb'))
    print('Permutation threshold - {}'.format(P.get_thresh(95)))
    print(P.get_obs_clusters(fl_mean_data)[2])

    lemon_plotting.plot_sensorspace_clusters(fl_mean_data, P, raw, ax1,
                                             xvect=freq_vect, base=0.5, thresh=97.5)
    ax1.set_ylabel('pseudo-t statistic')
    if subpanel is not None:
        lemon_plotting.subpanel_label(ax1, subpanel)
    title = p_name if title is None else title
    ax1.set_title('{0}'.format(title))

    fx = lemon_plotting.prep_scaled_freq(0.5, freq_vect)

    # we have already quantified the group means - plot them directly
    if p_name == 'Mean_OpenClosed':
        ax2.plot(fx[0], gmodel.copes[0, 2, :, :].mean(axis=1))
        ax2.plot(fx[0], gmodel.copes[0, 3, :, :].mean(axis=1))
    elif p_name == 'YoungOld_Mean':
        ax2.plot(fx[0], gmodel.copes[2, 0, :, :].mean(axis=1))
        ax2.plot(fx[0], gmodel.copes[3, 0, :, :].mean(axis=1))
    elif p_name == 'YoungOld_OpenClosed':
        ax2.plot(fx[0], gmodel.copes[2, 4, :, :].mean(axis=1))
        ax2.plot(fx[0], gmodel.copes[3, 4, :, :].mean(axis=1))
    else:
        # Model projected spectrum
        tags = ['OpenClosed', 'Linear', 'Sex', 'Brain', 'Grey']
        if np.any([p_name.find(tag) > -1 for tag in tags]):
            scales = [-1, 1]
        elif p_name.find('YoungOld') > -1:
            scales = [young_prop, -1-young_prop]
        else:
            scales = [0, 1]

        ix = np.ix_((0, 1), (0, 1, 2))
        beta0 = gmodel.copes[0, 0, :, :]
        print(beta0.shape)

        ix = np.ix_(np.atleast_1d(gl_con), np.atleast_1d(fl_con))
        beta1 = gmodel.copes[ix[0], ix[1], :, :].mean(axis=(0, 1))
        print(beta1.shape)

        ax2.plot(fx[0], np.mean(beta0 + scales[0]*beta1, axis=1))
        ax2.plot(fx[0], np.mean(beta0 + scales[1]*beta1, axis=1))

    ax2.set_xticks(fx[2], fx[1])
    lemon_plotting.decorate_spectrum(ax2)

#%% -----------------------------------------------------
# group overall design summary

sensor = 'Pz'
ch_ind = mne.pick_channels(raw.ch_names, [sensor])[0]

I = np.argsort(data.data[:, 0, :, 23].sum(axis=1))
I = np.arange(48)

plt.figure(figsize=(16, 9))
aspect = 16/9
xf = lemon_plotting.prep_scaled_freq(0.5, freq_vect)

subj_ax = plt.axes([0.05, 0.125, 0.35, 0.75])
des_ax = plt.axes([0.425, 0.1, 0.25, 0.75])
mean_ax = plt.axes([0.75, 0.6, 0.23, 0.25])
cov_ax = plt.axes([0.75, 0.1, 0.23, 0.25])

xstep = 35
ystep = 2e-6
ntotal = 36
subj_ax.plot((0, xstep*ntotal), (0, ystep*ntotal), color=[0.8, 0.8, 0.8], lw=0.5)
for ii in range(28):
    d = data.data[I[ii], 0, :, ch_ind]
    ii = ii + 8 if ii > 14 else ii
    subj_ax.plot(np.arange(len(freq_vect))+xstep*ii, d + ystep*ii)
for tag in ['top', 'right']:
    subj_ax.spines[tag].set_visible(False)
subj_ax.spines['bottom'].set_bounds(0, len(freq_vect))
subj_ax.spines['left'].set_bounds(0, 1e-5)
subj_ax.set_xlim(0)
subj_ax.set_ylim(0)
subj_ax.set_xticks([])
subj_ax.set_yticks([])
l = subj_ax.set_xlabel(r'Frequency (Hz) $\rightarrow$', loc='left')
l = subj_ax.set_ylabel(r'Amplitude $\rightarrow$', loc='bottom')
subj_ax.text(48+35*18, ystep*19, '...', fontsize='xx-large', rotation=52)
subj_ax.text(48+35*18, ystep*16, r'Participants $\rightarrow$', rotation=52)
lemon_plotting.subpanel_label(subj_ax, chr(65), yf=0.75, xf=0.05)
subj_ax.text(0.125, 0.725, 'First Level GLM\nCope-Spectra',
             transform=subj_ax.transAxes, fontsize='large')

with mpl.rc_context({'font.size': 7}):
    fig = glm.viz.plot_design_summary(design.design_matrix, design.regressor_names,
                                      contrasts=design.contrasts,
                                      contrast_names=design.contrast_names,
                                      ax=des_ax)
    fig.axes[4].set_position([0.685, 0.4, 0.01, 0.2])
    lemon_plotting.subpanel_label(fig.axes[1], chr(65+1), yf=1.1)
    fig.axes[1].set_ylabel('Participants')

mean_ax.errorbar(xf[0], gmodel.copes[0, 0, :, ch_ind],
                 yerr=np.sqrt(gmodel.varcopes[0, 0, :, ch_ind]), errorevery=1)
mean_ax.set_xticks(xf[2], xf[1])
mean_ax.set_title('Group Mean Spectrum')
lemon_plotting.decorate_spectrum(mean_ax, ylabel='Magnitude')
lemon_plotting.subpanel_label(mean_ax, chr(65+2), yf=1.1)
mean_ax.set_ylim(0)

cov_ax.errorbar(xf[0], gmodel.copes[1, 0, :, ch_ind],
                yerr=np.sqrt(gmodel.varcopes[1, 0, :, ch_ind]), errorevery=2)
cov_ax.errorbar(xf[0], gmodel.copes[4, 0, :, ch_ind],
                yerr=np.sqrt(gmodel.varcopes[4, 0, :, ch_ind]), errorevery=2)
cov_ax.errorbar(xf[0], gmodel.copes[5, 0, :, ch_ind],
                yerr=np.sqrt(gmodel.varcopes[5, 0, :, ch_ind]), errorevery=2)
cov_ax.errorbar(xf[0], gmodel.copes[6, 0, :, ch_ind],
                yerr=np.sqrt(gmodel.varcopes[6, 0, :, ch_ind]), errorevery=2)
cov_ax.set_title('Group effects on Mean Spectrum')
cov_ax.legend(list(np.array(gmodel.contrast_names)[[1, 4, 5, 6]]), frameon=False)
cov_ax.set_xticks(xf[2], xf[1])
lemon_plotting.decorate_spectrum(cov_ax, ylabel='Magnitude')
lemon_plotting.subpanel_label(cov_ax, chr(65+3), yf=1.1)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-overview.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-overview_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

#%% ----------------------------
# Group first-level covariates

fx = lemon_plotting.prep_scaled_freq(0.5, freq_vect,)

plt.figure(figsize=(16, 12))

ax = plt.axes((0.075, 0.675, 0.3, 0.25))
lemon_plotting.plot_joint_spectrum(ax, gmodel.copes[0, 2, :, :], raw, freq_vect,
                                   base=0.5, freqs=[10, 22], topo_scale=None,
                                   title='Group Con: Mean\nFirstLevel Con: Eyes Open',
                                   ylabel='Magnitude')
lemon_plotting.subpanel_label(ax, 'A')

ax = plt.axes((0.45, 0.675, 0.3, 0.25))
lemon_plotting.plot_joint_spectrum(ax, gmodel.copes[0, 3, :, :], raw, freq_vect,
                                   base=0.5, freqs=[10, 22], topo_scale=None,
                                   title='Group Con: Mean\nFirstLevel Con: Eyes Closed',
                                   ylabel='Magnitude')
lemon_plotting.subpanel_label(ax, 'B')

ax = plt.axes([0.775, 0.7, 0.15, 0.15])
lemon_plotting.plot_channel_layout(ax, raw, size=100)

ll = [['Rec Start', 'Rec End'],
      ['Good Seg', 'Bad Seg'],
      ['Good Seg', 'Bad Seg'],
      ['Low V-EOG Activity', 'High V-EOG Activity'],
      ['Low H-EOG Activity', 'High H-EOG Activity']]

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.45, hspace=0.5)

to_plot = ['Mean_Linear', 'Mean_BadSeg', 'Mean_DiffBadSeg', 'Mean_VEOG', 'Mean_HEOG']
labels = ['Group Con: Mean\nFirstLevel Con: Linear Trend',
          'Group Con: Mean\nFirstLevel Con: Bad Segments',
          'Group Con: Mean\nFirstLevel Con: Diff Bad Segments',
          'Group Con: Mean\nFirstLevel Con: VEOG',
          'Group Con: Mean\nFirstLevel Con: HEOG']

for ii in range(5):
    ax1 = plt.subplot(4, 5, ii+11)
    ax2 = plt.subplot(4, 5, ii+16)
    plot_perm_column(clean_data, gmodel, to_plot[ii], ax1, ax2, title=labels[ii])
    plt.legend(ll[ii], frameon=False)
    lemon_plotting.subpanel_label(ax1, chr(65+ii+2))
    ax2.set_ylabel('Magnitude')
    ax2.set_ylim(0)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-meancov.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-meancov_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

#%% ----------------------------
# Group ANOVA type plot

plt.figure(figsize=(16, 9))

ax1 = plt.axes([0.075, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.075, 0.1, 0.175, 0.30])
plot_perm_column(clean_data, gmodel, 'Mean_OpenClosed', ax1, ax2, title='Group Con: Mean\nFirstLevel Con: Open>Closed')
plt.legend(['Eyes Open Rest', 'Eyes Closed Rest'], frameon=False)
lemon_plotting.subpanel_label(ax1, 'A')
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0)

ax1 = plt.axes([0.3125, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.3125, 0.1, 0.175, 0.3])
plot_perm_column(clean_data, gmodel, 'YoungOld_Mean', ax1, ax2, title='Group Con: Young>Old\nFirstLevel Con: Mean')
plt.legend(['Young', 'Old'], frameon=False)
lemon_plotting.subpanel_label(ax1, 'B')
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0)

ax1 = plt.axes([0.55, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.55, 0.1, 0.175, 0.3])
plot_perm_column(clean_data, gmodel, 'YoungOld_OpenClosed', ax1, ax2, title='Group Con: Young>Old\nFirstLevel Con: Open>Closed')
ax2.set_ylim(-1.55e-6, 0.7e-6)
plt.legend(['Young : Open>Closed', 'Old : Open>Closed'], frameon=False, loc=1)
ax2.set_ylabel('Magnitude')
lemon_plotting.subpanel_label(ax1, 'C')

ax = plt.axes([0.775, 0.7, 0.2, 0.2])
lemon_plotting.plot_channel_layout(ax, raw, size=100)

ax = plt.axes([0.8, 0.175, 0.175, 0.35])
plt.plot(fx[0], gmodel.copes[2, 2, :, :].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 2, :, :].mean(axis=1))
plt.plot(fx[0], gmodel.copes[2, 3, :, :].mean(axis=1))
plt.plot(fx[0], gmodel.copes[3, 3, :, :].mean(axis=1))
plt.legend(['{0} {1}'.format(gmodel.contrast_names[2], fl_contrast_names[2][:-10]),
            '{0} {1}'.format(gmodel.contrast_names[3], fl_contrast_names[2][:-10]),
            '{0} {1}'.format(gmodel.contrast_names[2], fl_contrast_names[3][:-10]),
            '{0} {1}'.format(gmodel.contrast_names[3], fl_contrast_names[3][:-10])], frameon=False)
lemon_plotting.decorate_spectrum(ax, ylabel='Magnitude')
ax.set_xticks(fx[2], fx[1])
lemon_plotting.subpanel_label(ax, 'D')
ax.set_ylim(0, 6.5e-6)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-anova.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-anova_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

#%% ----------------------------
# GROUP COVARIATES

plt.figure(figsize=(16, 9))

# Sex effect
ax1 = plt.axes([0.075, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.075, 0.1, 0.175, 0.30])
plot_perm_column(clean_data, gmodel, 'Sex_Mean', ax1, ax2, title='Group Con: Sex\nFirstLevel Con: Mean')
lemon_plotting.subpanel_label(ax1, 'A')
plt.legend(['Female', 'Male'], frameon=False)
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0)

# Head Size
ax1 = plt.axes([0.3125, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.3125, 0.1, 0.175, 0.3])
plot_perm_column(clean_data, gmodel, 'BrainVol_Mean', ax1, ax2, title='Group Con: TotalBrainVol\nFirstLevel Con: Mean')
lemon_plotting.subpanel_label(ax1, 'B')
plt.legend(['Small Head', 'Large Head'], frameon=False)
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0)

# GreyMatter
ax1 = plt.axes([0.55, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.55, 0.1, 0.175, 0.3])
plot_perm_column(clean_data, gmodel, 'GreyVol_Mean', ax1, ax2, title='Group Con: GreyMatterVol\nFirstLevel Con: Mean')
lemon_plotting.subpanel_label(ax1, 'C')
plt.legend(['Low Grey Matter Vol', 'High Grey Matter Vol'], frameon=False)
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0)

ax = plt.axes([0.775, 0.7, 0.2, 0.2])
lemon_plotting.plot_channel_layout(ax, raw, size=100)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-covariates.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-covariates_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)
