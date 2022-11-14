"""This script runs the first-level GLM-Spectra and prepares the group data.

The script can be set to preprocess just the single example subject by changing
'subj' in the main body.

If the group-level data collation is run, then all glm-spectrum datasets found
in the output dir are collated.

"""

import glob
import os

import glmtools as glm
import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import pandas as pd
import sails
from anamnesis import obj_from_hdf5file

from glm_config import cfg
from lemon_support import (get_eeg_data, lemon_make_bads_regressor,
                           lemon_make_blinks_regressor,
                           lemon_make_task_regressor, quick_plot_eog_epochs,
                           quick_plot_eog_icas)

plt.switch_backend('agg')

#%% --------------------------------------------------------------
# Functions for first level analysis


def run_first_level(fname, outdir):
    """Run the first-level GLM-Spectrum for a single datafile."""
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
    fout = os.path.join(outdir, '{subj_id}_blink-summary.png'.format(subj_id=subj_id))
    blink_vect, numblinks, evoked_blink = lemon_make_blinks_regressor(raw, figpath=fout)

    fout = os.path.join(outdir, '{subj_id}_icaeog-summary.png'.format(subj_id=subj_id))
    quick_plot_eog_icas(raw, ica, figpath=fout)

    fout = os.path.join(outdir, '{subj_id}_{0}.png'.format('{0}', subj_id=subj_id))
    quick_plot_eog_epochs(raw, figpath=fout)

    veog = np.abs(raw.get_data(picks='ICA-VEOG')[0, :])#**2
    #veog = veog > np.percentile(veog, 97.5)

    heog = np.abs(raw.get_data(picks='ICA-HEOG')[0, :])#**2
    #heog = heog > np.percentile(heog, 97.5)

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
    confs = {'Bad Segs': bads_raw,
             'Bad Segs Diff': bads_diff,
             'V-EOG': veog, 'H-EOG': heog}
    eo_val = np.round(np.sum(task == 1) / len(task), 3)
    ec_val = np.round(np.sum(task == -1) / len(task), 3)
    conts = [{'name': 'OverallMean', 'values': {'Constant': 1,
                                                'Eyes Open': eo_val,
                                                'Eyes Closed': ec_val}},
             {'name': 'RestMean', 'values': {'Eyes Open': 0.5, 'Eyes Closed': 0.5}},
             {'name': 'Eyes Open AbsEffect', 'values': {'Constant': 1, 'Eyes Open': 0.5}},
             {'name': 'Eyes Closed AbsEffect', 'values': {'Constant': 1, 'Eyes Closed': 0.5}},
             {'name': 'Open > Closed', 'values': {'Eyes Open': 1, 'Eyes Closed': -1}}]

    fs = raw.info['sfreq']

    # Reduced model - no confounds or covariates
    freq_vect0, copes0, varcopes0, extras0 = sails.stft.glm_periodogram(
        XX, axis=0, fit_constant=True,
        conditions=conds, contrasts=conts,
        nperseg=int(fs * 2), noverlap=int(fs),
        fmin=0.1, fmax=100, fs=fs,
        mode="magnitude", fit_method="glmtools",
    )
    model0, design0, data0 = extras0
    print(model0.contrast_names)
    print(model0.design_matrix.shape)

    fs = raw.info['sfreq']

    # Full model
    freq_vect, copes, varcopes, extras = sails.stft.glm_periodogram(
        XX, axis=0, fit_constant=True,
        conditions=conds, covariates=covs,
        confounds=confs, contrasts=conts,
        nperseg=int(fs * 2), noverlap=int(fs),
        fmin=0.1, fmax=100, fs=fs,
        mode="magnitude", fit_method="glmtools",
    )

    model, design, data = extras
    print(model.contrast_names)
    print(model.design_matrix.shape)

    data.info['dim_labels'] = ['Windows', 'Frequencies', 'Sensors']

    print('----')
    print('Reduced Model AIC : {0} - R2 : {1}'.format(model0.aic.mean(), model0.r_square.mean()))
    print('Full Model AIC : {0} - R2 : {1}'.format(model.aic.mean(), model.r_square.mean()))

    data.info['dim_labels'] = ['Windows', 'Frequencies', 'Sensors']

    hdfname = os.path.join(outdir, '{subj_id}_glm-data.hdf5'.format(subj_id=subj_id))
    if os.path.exists(hdfname):
        print('Overwriting previous results')
        os.remove(hdfname)
    with h5py.File(hdfname, 'w') as F:
        model.to_hdf5(F.create_group('model'))
        model0.to_hdf5(F.create_group('reduced_model'))
        design.to_hdf5(F.create_group('design'))
        design0.to_hdf5(F.create_group('reduced_design'))
        data.to_hdf5(F.create_group('data'))
        F.create_dataset('freq_vect', data=freq_vect)
        F.create_dataset('chlabels', data=chlabels)
        F.create_dataset('scan_duration', data=raw.times[-1])
        F.create_dataset('num_blinks', data=numblinks)

    fout = os.path.join(outdir, '{subj_id}_glm-design.png'.format(subj_id=subj_id))
    design.plot_summary(show=False, savepath=fout)
    fout = os.path.join(outdir, '{subj_id}_glm-efficiency.png'.format(subj_id=subj_id))
    design.plot_efficiency(show=False, savepath=fout)

    quick_plot_firstlevel(hdfname, raw.filenames[0])

    plt.close('all')


def quick_plot_firstlevel(hdfname, rawpath):
    """Make a summary plot for a first level GLM-Spectrum."""
    model = obj_from_hdf5file(hdfname, 'model')
    freq_vect = h5py.File(hdfname, 'r')['freq_vect'][()]

    tstat_args = {'varcope_smoothing': 'medfilt', 'window_size': 15, 'smooth_dims': 1}
    ts = model.get_tstats(**tstat_args)
    ts = model.copes

    plt.figure(figsize=(16, 9))
    for ii in range(13):
        ax = plt.subplot(3, 5, ii+1)
        ax.plot(freq_vect, ts[ii, :, :])
        ax.set_title(model.contrast_names[ii])

    outf = hdfname.replace('.hdf5', '_glmsummary.png')
    plt.savefig(outf, dpi=300)
    plt.close('all')


#%% ---------------------------------------------------------
# Select datasets to run

# Set to * 'sub-010060' to run single subject in paper examples
#        * 'all' to run everyone
#        * None to run nothing and skip to group preparation
subj = 'all'

# Whether to collate first level results into group file, set to false if only
# running a single subject
collate_first_levels = True

#%% -----------------------------------------------------------
#  Run first level GLMs

proc_outdir = cfg['lemon_processed_data']

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}_preproc_raw.fif')
st = osl.utils.Study(fbase)

if subj == 'all':
    inputs = st.match_files
else:
    inputs = st.get(subj=subj)

for fname in inputs:
    try:
        run_first_level(fname, cfg['lemon_glm_data'])
    except Exception as e:
        print(e)
        pass

#%% -------------------------------------------------------
# Run first-level model selection
# takes a while and independent from most results so run separately

print('Loading first levels')

# Get first level filenames
fnames = sorted(glob.glob(cfg['lemon_glm_data'] + '/sub*glm-data.hdf5'))

r2_group = []
# Main loop - load first levels and store copes + meta data
for idx, fname in enumerate(fnames):
    print('{0}/{1} - {2}'.format(idx, len(fnames), fname.split('/')[-1]))

    # Load data and design
    design = obj_from_hdf5file(fname, 'design')
    data = obj_from_hdf5file(fname, 'data')

    models = glm.fit.run_regressor_selection(design, data)

    r2 = np.concatenate([m.r_square[None, None, 0, :, :] * 100 for m in models], axis=1)
    r2_group.append(r2)

r2_group = np.concatenate(r2_group, axis=0)

outf = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_rsquared.npy')
np.save(outf, r2_group)



#%% -------------------------------------------------------
# Prepare first level files for group analysis

print('Loading first levels')

# Get first level filenames
fnames = sorted(glob.glob(cfg['lemon_glm_data'] + '/sub*glm-data.hdf5'))

# Load subject meta data
fname = 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
meta_file = os.path.join(os.path.dirname(cfg['lemon_raw'].rstrip('/')), fname)
df = pd.read_csv(meta_file)

# Extract subject IDs
allsubj = np.unique([fname.split('/')[-1].split('_')[0][4:] for fname in fnames])
allsubj_no = np.arange(len(allsubj))

# Preallocate lists
subj_id = []
subj = []
age = []
sex = []
hand = []
task = []
scandur = []
num_blinks = []

first_level = []
first_level_reduced = []
r2 = []
r2_reduced = []
aic = []
aic_reduced = []
sw = []
sw_reduced = []

# Main loop - load first levels and store copes + meta data
for idx, fname in enumerate(fnames):
    print('{0}/{1} - {2}'.format(idx, len(fnames), fname.split('/')[-1]))

    # Load data and design
    design = obj_from_hdf5file(fname, 'design')
    reduced_design = obj_from_hdf5file(fname, 'reduced_design')
    data = obj_from_hdf5file(fname, 'data')

    # Load reduced model
    reduced_model = obj_from_hdf5file(fname, 'reduced_model')
    reduced_model.design_matrix = reduced_design.design_matrix
    first_level_reduced.append(reduced_model.copes[None, :, :, :])
    r2_reduced.append(reduced_model.r_square.mean())
    sw_reduced.append(reduced_model.get_shapiro(data.data).mean())
    aic_reduced.append(reduced_model.aic.mean())

    # Load full model
    model = obj_from_hdf5file(fname, 'model')
    model.design_matrix = design.design_matrix
    first_level.append(model.copes[None, :, :, :])
    r2.append(model.r_square.mean())
    sw.append(model.get_shapiro(data.data).mean())
    aic.append(model.aic.mean())

    s_id = fname.split('/')[-1].split('_')[0][4:]
    subj.append(np.where(allsubj == s_id)[0][0])
    subj_id.append(s_id)
    if fname.find('EO') > 0:
        task.append(1)
    elif fname.find('EC') > 0:
        task.append(2)

    demo_ind = np.where(df['ID'].str.match('sub-' + s_id))[0]
    if len(demo_ind) > 0:
        tmp_age = df.iloc[demo_ind[0]]['Age']
        age.append(np.array(tmp_age.split('-')).astype(float).mean())
        sex.append(df.iloc[demo_ind[0]]['Gender_ 1=female_2=male'])
    num_blinks.append(h5py.File(fname, 'r')['num_blinks'][()])

# Stack first levels into new glm dataset +  save
first_level = np.concatenate(first_level, axis=0)
group_data = glm.data.TrialGLMData(data=first_level, subj_id=subj_id, shapiro=sw,
                                   subj=subj, task=task, age=age, num_blinks=num_blinks,
                                   sex=sex, scandur=scandur, aic=aic, r2=r2)

outf = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata.hdf5')
with h5py.File(outf, 'w') as F:
    group_data.to_hdf5(F.create_group('data'))
    F.create_dataset('aic', data=aic)
    F.create_dataset('r2', data=r2)

first_level_reduced = np.concatenate(first_level_reduced, axis=0)
group_data = glm.data.TrialGLMData(data=first_level_reduced, subj_id=subj_id, shapiro=sw_reduced,
                                   subj=subj, task=task, age=age, num_blinks=num_blinks,
                                   sex=sex, scandur=scandur, aic=aic_reduced, r2=r2_reduced)

outf = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata_reduced.hdf5')
with h5py.File(outf, 'w') as F:
    group_data.to_hdf5(F.create_group('data'))
    F.create_dataset('aic', data=aic_reduced)
    F.create_dataset('r2', data=r2_reduced)