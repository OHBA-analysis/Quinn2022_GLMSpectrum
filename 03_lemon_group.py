"""This script runs the group-level GLM-Spectra and non-parametric permutations.

The script can be set to preprocess just the single example subject by changing
'subj' in the main body.

If the group-level data collation is run, then all glm-spectrum datasets found
in the output dir are collated.

"""

import os
from copy import deepcopy

import dill
import glmtools as glm
import h5py
import mne
import numpy as np
import osl
import pandas as pd
import sails
from anamnesis import obj_from_hdf5file

from glm_config import cfg

#%% ---------------------------------------------------
# Load single subject for reference

fbase = os.path.join(cfg['lemon_processed_data'], 'sub-010002_preproc_raw.fif')
raw = mne.io.read_raw_fif(fbase).pick_types(eeg=True)

st = osl.utils.Study(os.path.join(cfg['lemon_glm_data'], '{subj}_preproc_raw_glm-data.hdf5'))
freq_vect = h5py.File(st.match_files[0], 'r')['freq_vect'][()]
fl_model = obj_from_hdf5file(st.match_files[0], 'model')

df = pd.read_csv(os.path.join(cfg['code_dir'], 'lemon_structural_vols.csv'))

#%% --------------------------------------------------
# Load first level results and fit group model

inputs = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata.hdf5')
reduceds = os.path.join(cfg['lemon_glm_data'], 'lemon_eeg_sensorglm_groupdata_reduced.hdf5')

data = obj_from_hdf5file(inputs, 'data')
datareduced = obj_from_hdf5file(reduceds, 'data')

data.info['age_group'] = np.array(data.info['age']) < 45

tbv = []
gmv = []
htv = []
for subj_id in data.info['subj_id']:
    row = df[df['PPT_ID'] == 'sub-' + subj_id]
    if len(row) > 0:
        tbv.append(row[' TotalBrainVol'].values[0])
        gmv.append(row[' GreyMatterVolNorm'].values[0])
        htv.append(row[' HippoTotalVolNorm'].values[0])
    else:
        tbv.append(np.nan)
        gmv.append(np.nan)
        htv.append(np.nan)

data.info['total_brain_vol'] = np.array(tbv)
data.info['grey_matter_vol'] = np.array(gmv)
data.info['hippo_vol'] = np.array(htv)
datareduced.info['total_brain_vol'] = np.array(tbv)
datareduced.info['grey_matter_vol'] = np.array(gmv)
datareduced.info['hippo_vol'] = np.array(htv)

# Drop obvious outliers & those with missing MR
bads = sails.utils.detect_artefacts(data.data[:, 0, :, :], axis=0)
bads = np.logical_or(bads, np.isnan(htv))
clean_data = data.drop(np.where(bads)[0])
clean_reduced = datareduced.drop(np.where(bads)[0])

DC = glm.design.DesignConfig()
DC.add_regressor(name='Young', rtype='Categorical', datainfo='age_group', codes=1)
DC.add_regressor(name='Old', rtype='Categorical', datainfo='age_group', codes=0)
DC.add_regressor(name='Sex', rtype='Parametric', datainfo='sex', preproc='z')
DC.add_regressor(name='TotalBrainVol', rtype='Parametric', datainfo='total_brain_vol', preproc='z')
DC.add_regressor(name='GreyMatterVol', rtype='Parametric', datainfo='grey_matter_vol', preproc='z')

young_prop = np.round(np.sum(np.array(data.info['age']) < 40) / len(data.info['age']), 3)
old_prop = np.round(np.sum(np.array(data.info['age']) > 40) / len(data.info['age']), 3)

DC.add_contrast(name='Mean', values={'Young': young_prop, 'Old': old_prop})
DC.add_contrast(name='Young>Old', values={'Young': 1, 'Old': -1})
DC.add_simple_contrasts()

design = DC.design_from_datainfo(clean_data.info)
gmodel = glm.fit.OLSModel(design, clean_data)
gmodel_reduced = glm.fit.OLSModel(design, clean_reduced)

with h5py.File(os.path.join(cfg['lemon_glm_data'], 'lemon-group_glm-data.hdf5'), 'w') as F:
    gmodel.to_hdf5(F.create_group('model'))
    design.to_hdf5(F.create_group('design'))
    # hdf5 is messy sometimes - hopefully someone improves string type handling sometime
    clean_data.info['subj_id'] = np.array(clean_data.info['subj_id'],
                                          dtype=h5py.special_dtype(vlen=str))
    clean_data.to_hdf5(F.create_group('data'))

fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-design.png')
design.plot_summary(show=False, savepath=fout)
fout = os.path.join(cfg['lemon_figures'], 'lemon-group_glm-efficiency.png')
design.plot_efficiency(show=False, savepath=fout)

fl_contrast_names = ['OverallMean', 'RestMean',
                     'Eyes Open AbsEffect', 'Eyes Closed AbsEffect', 'Open > Closed',
                     'Constant', 'Eyes Open', 'Eyes Closed',
                     'Linear Trend', 'Bad Segs', 'Bad Segs Diff', 'V-EOG', 'H-EOG']

#%% ------------------------------------------------------
# Permutation stats - run or load from disk

adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(raw.info, 'eeg')
ntests = np.prod(data.data.shape[2:])
ntimes = data.data.shape[2]
adjacency = mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)

cft = 3
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

for icon in range(len(to_permute)):
    gl_con, fl_con, gl_reg, fl_reg, p_name = to_permute[icon]
    print('### Contrast - {}'.format(p_name))
    print('permuting')
    # Only working with mean regressor for the moment
    fl_mean_data = deepcopy(clean_data)
    fl_mean_data.data = clean_data.data[:, fl_con, :, :]

    if icon > 0 and gl_con == 0:
        cft = 6
    else:
        cft = 3

    p = glm.permutations.MNEClusterPermutation(design, fl_mean_data, gl_con, 1500,
                                               nprocesses=8,
                                               metric='tstats',
                                               cluster_forming_threshold=cft,
                                               tstat_args=tstat_args,
                                               adjacency=adjacency)

    dill_fname = os.path.join(cfg['lemon_glm_data'], 'lemon-group_perms-{0}.pkl'.format(p_name))
    with open(dill_fname, "wb") as dill_file:
        dill.dump(p, dill_file)
