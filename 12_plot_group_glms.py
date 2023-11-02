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
from matplotlib.lines import Line2D
from lemon_support import lemon_set_standard_montage

import lemon_plotting
from glm_config import cfg

#%% ---------------------------------------------------
# Load single subject for reference

perm_name = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{pname}.pkl')
perms = osl.utils.Study(perm_name)

fl_path = os.path.join(cfg['lemon_glm_data'],'{subj_id}_preproc_raw-glm-spectrum_model-{model}_data.pkl')
st = osl.utils.Study(fl_path)
glmsp = osl.glm.read_glm_spectrum(st.get(model='full')[0])

gl_name = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-{model}.pkl')
glglms = osl.utils.Study(gl_name)
fglmsp = osl.glm.read_glm_spectrum(glglms.get(model='full_data')[0])
rglmsp = osl.glm.read_glm_spectrum(glglms.get(model='reduced_data')[0])

rglmsp = lemon_set_standard_montage(rglmsp)
fglmsp = lemon_set_standard_montage(fglmsp)


#%% ------------------------------------------------------

flcf = np.load(os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_fl-cohens-f2.npy'))
flr2 = np.load(os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_fl-r-square.npy'))
glcf = np.load(os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_gl-cohens-f2.npy'))
glr2 = np.load(os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_gl-r-square.npy'))

fx, ftl, ft = lemon_plotting.prep_scaled_freq(0.5, fglmsp.f)

lc = [0.8, 0.8, 0.8]

plt.figure(figsize=(12, 9))
# First levels
ax = plt.axes([0.075, 0.6, 0.2, 0.35])
ax.plot(fx, flr2[:, 0, :, :].mean(axis=1).T, lw=0.5, color=lc)
ax.plot(fx, flr2[:, 0, :, :].mean(axis=(0, 1)), lw=2, color='k')
ax.set_xticks(ft)
ax.set_xticklabels(ftl)
lemon_plotting.decorate_spectrum(ax, ylabel="R-Square")
ax.set_title('Full Model', y=0.95)
lines = [Line2D([0], [0], color=lc, linewidth=0.5), Line2D([0], [0], color='k', linewidth=2)]
plt.legend(lines, ['Single Dataset', 'Group Average'])

ax = plt.axes([0.35, 0.6, 0.2, 0.35])
ax.plot(fx, flcf[:, 1, :, :].mean(axis=1).T, lw=0.5, color=lc)
ax.plot(fx, flcf[:, 1, :, :].mean(axis=(0, 1)), lw=2, color='k')
ax.set_xticks(ft)
ax.set_xticklabels(ftl)
lemon_plotting.decorate_spectrum(ax, ylabel="Cohen's F2")
ax.set_title('Eyes Open > Eyes Closed', y=0.95)

axs = [plt.axes([0.6, 0.8, 0.18, 0.18]), plt.axes([0.8, 0.8, 0.18, 0.18]),
       plt.axes([0.6, 0.55, 0.18, 0.18]), plt.axes([0.8, 0.55, 0.18, 0.18])]

for idx, ax in enumerate(axs):
    ax.plot(fx, flcf[:, idx+2, :, :].mean(axis=1).T, lw=0.5, color=lc)
    ax.plot(fx, flcf[:, idx+2, :, :].mean(axis=(0, 1)), lw=2, color='k')
    ax.set_xticks(ft)
    ax.set_xticklabels(ftl)
    ax.set_title(glmsp.model.regressor_names[idx+2], y=0.85)
    ax.set_ylim(0, 0.5)
    lemon_plotting.decorate_spectrum(ax, ylabel="Cohens F2")
    if idx < 2:
        ax.set_xlabel('')
    if idx == 1 or idx == 3:
        ax.set_ylabel('')

ff = [(5, 17), (6.5, 17, 45), (8, 16), (5, 10)]
for ii in range(4):
    ax = plt.axes([0.05+0.24*ii, 0.1, 0.175, 0.35])
    yl = "Cohen's F2" if ii == 0 else ""
    title = "Age" if ii == 0 else fglmsp.model.regressor_names[ii+1]
    osl.glm.plot_joint_spectrum(fglmsp.f, glcf[ii+1, 0, :, :].T, glmsp.info, ax=ax, base=0.5,
                                ylabel=yl, title=title, freqs=ff[ii])
    ax.set_ylim(0, 0.5)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_model-stat-summary.png')
plt.savefig(fout, transparent=True, dpi=300)   
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_model=stat-summary_low-res.png')
plt.savefig(fout, transparent=True, dpi=100) 

#%% ------------------------------------------------------
# Group-level experimental difference between null and full models

diff = fglmsp.model.tstats[0, 1, :, :] - rglmsp.model.tstats[0, 1, :, :]
freqs = (2, 8, 23)
plt.figure(figsize=(12, 5))
plt.subplots_adjust(left=0.05, right=0.95, hspace=0.3)
ax = plt.subplot(131)
t = 'Eyes Open > Eyes Closed\nNo Covariates'
rglmsp.plot_joint_spectrum(0, 1, ax=ax, base=0.5, ylabel='t-stats', metric='tstats', title=t, freqs=freqs)
lemon_plotting.subpanel_label(ax, chr(65))

ax = plt.subplot(132)
t = 'Eyes Open > Eyes Closed\nWith Covariates'
fglmsp.plot_joint_spectrum(0, 1, ax=ax, base=0.5, ylabel='t-stats', metric='tstats', title=t, freqs=freqs)
lemon_plotting.subpanel_label(ax, chr(66))

ax = plt.subplot(133)
osl.glm.plot_joint_spectrum(fglmsp.f, diff.T, fglmsp.info, ax=ax, base=0.5, ylabel='t-stat difference', title='Difference', freqs=freqs)
lemon_plotting.subpanel_label(ax, chr(67))

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_confound-openclosed.png')
plt.savefig(fout, transparent=True, dpi=300)   
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_confound-openclosed_low-res.png')
plt.savefig(fout, transparent=True, dpi=100) 


#%% ------------------------------------------------------
# Effect of covariates across first-levels

fx, ftl, ft = lemon_plotting.prep_scaled_freq(0.5, fglmsp.f)

lc = [0.8, 0.8, 0.8]
ppts = [43, 145, 68, 62, 112]
ppts = ['sub-010060', 'sub-010255', 'sub-010086', 'sub-010079', 'sub-010213']
fnames = sorted(st.get())
ppts2 = []
for idx in range(len(ppts)):
    tmp = np.where([fn.find(ppts[idx]) > -1 for fn in fnames])
    ppts2.append(tmp[0][0])
ppts2[1] -= 1
ppts2[2] -= 2
ppts2[3] += 2
ppts2[4] -= 3
ppts = ppts2

labs = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
fontargs = {'fontsize': 'large', 'fontweight': 'bold', 'color': 'r', 'ha': 'center', 'va': 'center'}

plt.figure(figsize=(16, 9))

ax = plt.axes([0.075, 0.55, 0.2, 0.35])
ax.plot(fx, flr2[:, 0, :, :].mean(axis=1).T, lw=0.5, color=lc)
ax.plot(fx, flr2[:, 0, :, :].mean(axis=(0, 1)), lw=2, color='k')
ax.set_xticks(ft)
ax.set_xticklabels(ftl)
lemon_plotting.decorate_spectrum(ax, ylabel="R-Square")
ax.set_title('Full Model', y=0.95)
lines = [Line2D([0], [0], color=lc, linewidth=0.5), Line2D([0], [0], color='k', linewidth=2)]
plt.legend(lines, ['Single Dataset', 'Group Average'])
lemon_plotting.subpanel_label(ax, chr(65), yf=1.1)

ax = plt.axes([0.35, 0.55, 0.2, 0.35])
ax.plot(fx, flcf[:, 1, :, :].mean(axis=1).T, lw=0.5, color=lc)
ax.plot(fx, flcf[:, 1, :, :].mean(axis=(0, 1)), lw=2, color='k')
ax.set_xticks(ft)
ax.set_xticklabels(ftl)
lemon_plotting.decorate_spectrum(ax, ylabel="Cohen's F2")
ax.set_title('Eyes Open > Eyes Closed', y=0.95)
lemon_plotting.subpanel_label(ax, chr(65+1), yf=1.1)

axs = [plt.axes([0.6, 0.75, 0.18, 0.18]), plt.axes([0.8, 0.75, 0.18, 0.18]),
       plt.axes([0.6, 0.5, 0.18, 0.18]), plt.axes([0.8, 0.5, 0.18, 0.18])]

for idx, ax in enumerate(axs):
    ax.plot(fx, flcf[:, idx+2, :, :].mean(axis=1).T, lw=0.5, color=lc)
    ax.plot(fx, flcf[:, idx+2, :, :].mean(axis=(0, 1)), lw=2, color='k')
    ax.set_xticks(ft)
    ax.set_xticklabels(ftl)
    ax.set_title(glmsp.model.regressor_names[idx+2], y=0.85)
    ax.set_ylim(0, 0.5)
    lemon_plotting.decorate_spectrum(ax, ylabel="Cohen's F2")
    if idx < 2:
        ax.set_xlabel('')
    if idx == 1 or idx == 3:
        ax.set_ylabel('')
    if idx == 0:
        lemon_plotting.subpanel_label(ax, chr(65+2), yf=1.1)


labs = ['D i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii']
for ii in range(len(ppts[:5])):
    ax = plt.axes((0.1+ii*0.18, 0.07, 0.125, 0.3))
    ax.plot(fx, rglmsp.data.data[ppts[ii], 0, :, :].mean(axis=0))
    ax.plot(fx, fglmsp.data.data[ppts[ii], 0, :, :].mean(axis=0))
    ax.set_xticks(ft)[::2]
    ax.set_xticklabels(ftl)[::2]
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    #plt.ylim(0, 1.2e-5)
    lemon_plotting.subpanel_label(plt.gca(), labs[ii])
    if ii == 0:
        plt.ylabel('Magnitude')
    plt.xlabel('Frequency (Hz)')
plt.legend(['reduced Model', 'Full Model'], frameon=False)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_stats-summary.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_stats-summary_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)


#%% -----------------------------------------------------
# group overall design summary

sensor = 'Pz'
ch_ind = mne.pick_channels(glmsp.info['ch_names'], [sensor])[0]

I = np.arange(48)

plt.figure(figsize=(16, 9))
aspect = 16/9
xf = lemon_plotting.prep_scaled_freq(0.5, fglmsp.f)

subj_ax = plt.axes([0.025, 0.125, 0.35, 0.75])
des_ax = plt.axes([0.4, 0.1, 0.25, 0.75])
mean_ax = plt.axes([0.725, 0.55, 0.25, 0.35])
cov_ax = plt.axes([0.725, 0.1, 0.25, 0.35])

xstep = 35
ystep = 2e-6
ntotal = 36
subj_ax.plot((0, xstep*ntotal), (0, ystep*ntotal), color=[0.8, 0.8, 0.8], lw=0.5)
for ii in range(28):
    d = fglmsp.data.data[I[ii], 0, ch_ind, :]
    ii = ii + 8 if ii > 14 else ii
    subj_ax.plot(np.arange(len(xf[0]))+xstep*ii, d + ystep*ii)
for tag in ['top', 'right']:
    subj_ax.spines[tag].set_visible(False)
subj_ax.spines['bottom'].set_bounds(0, len(xf[0]))
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
    fig = glm.viz.plot_design_summary(fglmsp.design.design_matrix, fglmsp.design.regressor_names,
                                      contrasts=fglmsp.design.contrasts,
                                      contrast_names=fglmsp.design.contrast_names,
                                      ax=des_ax)
    fig.axes[4].set_position([0.655, 0.4, 0.01, 0.2])
    lemon_plotting.subpanel_label(fig.axes[1], chr(65+1), yf=1.1)
    fig.axes[1].set_ylabel('Participants')

mean_ax.errorbar(xf[0], fglmsp.model.copes[0, 0, ch_ind, :],
                 yerr=np.sqrt(fglmsp.model.varcopes[0, 0, ch_ind, :]), errorevery=1)
mean_ax.set_xticks(xf[2], xf[1])
mean_ax.set_title('Group Mean Spectrum')
lemon_plotting.decorate_spectrum(mean_ax, ylabel='Magnitude')
lemon_plotting.subpanel_label(mean_ax, chr(65+2), yf=1.1)
mean_ax.set_ylim(0)

cov_ax.errorbar(xf[0], fglmsp.model.copes[1, 0, ch_ind, :],
                yerr=np.sqrt(fglmsp.model.varcopes[1, 0, ch_ind, :]), errorevery=2)
cov_ax.errorbar(xf[0], fglmsp.model.copes[4, 0, ch_ind, :],
                yerr=np.sqrt(fglmsp.model.varcopes[4, 0, ch_ind, :]), errorevery=2)
cov_ax.errorbar(xf[0], fglmsp.model.copes[5, 0, ch_ind, :],
                yerr=np.sqrt(fglmsp.model.varcopes[5, 0, ch_ind, :]), errorevery=2)
cov_ax.errorbar(xf[0], fglmsp.model.copes[6, 0, ch_ind, :],
                yerr=np.sqrt(fglmsp.model.varcopes[6, 0, ch_ind, :]), errorevery=2)
cov_ax.set_title('Group effects on Mean Spectrum')
cov_ax.legend(list(np.array(fglmsp.model.contrast_names)[[1, 4, 5, 6]]), frameon=False, fontsize=12)
cov_ax.set_xticks(xf[2], xf[1])
lemon_plotting.decorate_spectrum(cov_ax, ylabel='Magnitude')
lemon_plotting.subpanel_label(cov_ax, chr(65+3), yf=1.1)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-overview.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-overview_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)


fout = os.path.join(cfg['lemon_figures'], 'lemon-group_grouplevel-design.png')
fig = fglmsp.design.plot_summary(figargs={'figsize': (16, 12)}, show=False)
fig.savefig(fout, dpi=100, transparent=True)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_grouplevel-efficiency.png')
fig = fglmsp.design.plot_efficiency()
fig.savefig(fout, dpi=100, transparent=True)

eye

#%% ----------------------------
# Group first-level covariates

def project_first_level_range(glmsp, contrast, nsteps=2, min_step=0, max_step=1):
    steps = np.linspace(min_step, max_step, nsteps)
    pred = np.zeros((nsteps, *glmsp.model.betas.shape[2:]))

    # Run projection
    for ii in range(nsteps):
        if nsteps == 1:
            coeff = 0
        else:
            coeff = steps[ii]
        pred[ii, ...] = glmsp.model.betas[0, 0, ...] + coeff*glmsp.model.betas[0, contrast, ...]

    return pred

fx = lemon_plotting.prep_scaled_freq(0.5, fglmsp.f,)

plt.figure(figsize=(16, 9))

ax = plt.axes([0.775, 0.7, 0.15, 0.15])
lemon_plotting.plot_channel_layout2(ax, fglmsp.info, size=100)

ll = [['Rec Start', 'Rec End'],
      ['Good Seg', 'Bad Seg'],
      ['Low V-EOG Activity', 'High V-EOG Activity'],
      ['Low H-EOG Activity', 'High H-EOG Activity']]

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.45, hspace=0.5, top=0.9)

perms = ['MeanLinear Trend', 'MeanBad Segs', 'MeanV-EOG', 'MeanH-EOG']

for ii in range(4):
    ax1 = plt.subplot(3, 4, (ii+1, ii+5))
    Pname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{}.pkl'.format(perms[ii]))
    P = lemon_set_standard_montage(osl.glm.read_glm_spectrum(Pname))
    P.plot_sig_clusters(95, ax=ax1, base=0.5)
    plt.legend(ll[ii], frameon=False)
    lemon_plotting.subpanel_label(ax1, chr(65+ii), yf=0.75)
    ax2 = plt.subplot(3, 4, ii+9)
    if ii == 0:
        proj = project_first_level_range(fglmsp, 4+ii, min_step=-1.73, max_step=1.73)
    else:
        proj = project_first_level_range(fglmsp, 4+ii)
    ax2.plot(fx[0], proj.mean(axis=1).T)
    ax2.set_xticks(fx[2])
    ax2.set_xticklabels(fx[1])
    ax2.set_xlim(fx[0][0], fx[0][-1])
    lemon_plotting.decorate_spectrum(ax2)
    ax2.legend(ll[ii], frameon=False)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-meancov.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-meancov_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

#%% ----------------------------
# Group ANOVA type plot

fx, ftl, ft = lemon_plotting.prep_scaled_freq(0.5, fglmsp.f)

plt.figure(figsize=(16, 9))

ax1 = plt.axes([0.075, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.075, 0.1, 0.175, 0.30])
Pname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{}.pkl'.format('MeanOpen>Closed'))
P = lemon_set_standard_montage(osl.glm.read_glm_spectrum(Pname))
P.plot_sig_clusters(95, ax=ax1, base=0.5)
lemon_plotting.subpanel_label(ax1, 'A')

ax2.plot(fx, fglmsp.model.copes[0, 2, :, :].mean(axis=0))
ax2.plot(fx, fglmsp.model.copes[0, 3, :, :].mean(axis=0))
lemon_plotting.decorate_spectrum(ax2)
ax2.set_xticks(ft)
ax2.set_xticklabels(ftl)
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0)
plt.legend(['Eyes Open Rest', 'Eyes Closed Rest'], frameon=False)


ax1 = plt.axes([0.3125, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.3125, 0.1, 0.175, 0.3])
Pname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{}.pkl'.format('Young>OldRestMean'))
P = lemon_set_standard_montage(osl.glm.read_glm_spectrum(Pname))
P.plot_sig_clusters(95, ax=ax1, base=0.5)
lemon_plotting.subpanel_label(ax1, 'B')
ax2.plot(fx, fglmsp.model.copes[2, 0, :, :].mean(axis=0))
ax2.plot(fx, fglmsp.model.copes[3, 0, :, :].mean(axis=0))
lemon_plotting.decorate_spectrum(ax2)
ax2.set_xticks(ft)
ax2.set_xticklabels(ftl)
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0)
plt.legend(['Young', 'Old'], frameon=False)


ax1 = plt.axes([0.55, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.55, 0.1, 0.175, 0.3])
Pname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{}.pkl'.format('Young>OldOpen>Closed'))
P = lemon_set_standard_montage(osl.glm.read_glm_spectrum(Pname))
P.plot_sig_clusters(95, ax=ax1, base=0.5)
lemon_plotting.subpanel_label(ax1, 'C')
ax2.plot(fx, fglmsp.model.copes[2, 1, :, :].mean(axis=0))
ax2.plot(fx, fglmsp.model.copes[3, 1, :, :].mean(axis=0))
lemon_plotting.decorate_spectrum(ax2)
ax2.set_xticks(ft)
ax2.set_xticklabels(ftl)
ax2.set_ylabel('Magnitude')
plt.legend(['YoungOpen>Closed', 'OldOpen>Closed'], frameon=False)

ax = plt.axes([0.775, 0.7, 0.2, 0.2])
lemon_plotting.plot_channel_layout2(ax, fglmsp.info, size=100)

ax = plt.axes([0.8, 0.175, 0.175, 0.35])
plt.plot(fx, fglmsp.model.copes[2, 2, :, :].mean(axis=0))
plt.plot(fx, fglmsp.model.copes[3, 2, :, :].mean(axis=0))
plt.plot(fx, fglmsp.model.copes[2, 3, :, :].mean(axis=0))
plt.plot(fx, fglmsp.model.copes[3, 3, :, :].mean(axis=0))
plt.legend(['{0} {1}'.format(fglmsp.model.contrast_names[2], fglmsp.fl_contrast_names[2][:-10]),
            '{0} {1}'.format(fglmsp.model.contrast_names[3], fglmsp.fl_contrast_names[2][:-10]),
            '{0} {1}'.format(fglmsp.model.contrast_names[2], fglmsp.fl_contrast_names[3][:-10]),
            '{0} {1}'.format(fglmsp.model.contrast_names[3], fglmsp.fl_contrast_names[3][:-10])], frameon=False)
lemon_plotting.decorate_spectrum(ax, ylabel='Magnitude')
ax2.set_xticks(ft)
ax2.set_xticklabels(ftl)
ax.set_ylim(0, 6.5e-6)
lemon_plotting.subpanel_label(ax, 'D')

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
Pname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{}.pkl'.format('SexRestMean'))
P = lemon_set_standard_montage(osl.glm.read_glm_spectrum(Pname))
P.plot_sig_clusters(95, ax=ax1, base=0.5)
lemon_plotting.subpanel_label(ax1, 'C')
proj, ll = fglmsp.model.project_range(2, nsteps=2)
ax2.plot(fx, proj[0, 0, :, :].mean(axis=0))
ax2.plot(fx, proj[1, 0, :, :].mean(axis=0))
lemon_plotting.decorate_spectrum(ax2)
ax2.set_xticks(ft)
ax2.set_xticklabels(ftl)
ax2.set_ylabel('Magnitude')
plt.legend(['Female', 'Male'], frameon=False)


# Head Size
ax1 = plt.axes([0.3125, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.3125, 0.1, 0.175, 0.3])
Pname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{}.pkl'.format('TotalBrainVolRestMean'))
P = lemon_set_standard_montage(osl.glm.read_glm_spectrum(Pname))
P.plot_sig_clusters(95, ax=ax1, base=0.5)
lemon_plotting.subpanel_label(ax1, 'C')
proj, ll = fglmsp.model.project_range(3, nsteps=2)
ax2.plot(fx, proj[0, 0, :, :].mean(axis=0))
ax2.plot(fx, proj[1, 0, :, :].mean(axis=0))
lemon_plotting.decorate_spectrum(ax2)
ax2.set_xticks(ft)
ax2.set_xticklabels(ftl)
ax2.set_ylabel('Magnitude')
plt.legend(['Small Head', 'Large Head'], frameon=False)

# GreyMatter
ax1 = plt.axes([0.55, 0.5, 0.175, 0.4])
ax2 = plt.axes([0.55, 0.1, 0.175, 0.3])
Pname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-spec_model-full_perm-{}.pkl'.format('GreyMatterVolRestMean'))
P = lemon_set_standard_montage(osl.glm.read_glm_spectrum(Pname))
P.plot_sig_clusters(95, ax=ax1, base=0.5)
lemon_plotting.subpanel_label(ax1, 'C')
proj, ll = fglmsp.model.project_range(4, nsteps=2)
ax2.plot(fx, proj[0, 0, :, :].mean(axis=0))
ax2.plot(fx, proj[1, 0, :, :].mean(axis=0))
lemon_plotting.decorate_spectrum(ax2)
ax2.set_xticks(ft)
ax2.set_xticklabels(ftl)
ax2.set_ylabel('Magnitude')
plt.legend(['Small Vol', 'Large Vol'], frameon=False)

ax = plt.axes([0.725, 0.7, 0.2, 0.2])
lemon_plotting.plot_channel_layout2(ax, fglmsp.info, size=100)

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-covariates.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_group-glm-covariates_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)
