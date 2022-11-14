"""This script produces a table summarising the preprocessing outcomes."""

import os

import mne
import numpy as np
import osl
import pandas as pd

from glm_config import cfg

#%% --------------------------------------------------------------
# Find preprocessed data

proc_outdir = cfg['lemon_processed_data']

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}_{ftype}.fif')
st = osl.utils.Study(fbase)

#%% --------------------------------------------------------------
# Find preprocessed data

preprocs = []

for subj_id in np.unique(st.fields['subj']):
    print(subj_id)
    raw = mne.io.read_raw_fif(st.get(subj=subj_id, ftype='preproc_raw')[0])
    ica = mne.preprocessing.read_ica(st.get(subj=subj_id, ftype='ica')[0])

    # We later mark the very start and end as bad - so the counters don't start at zero
    bads = 6
    dbads = 2
    for an in raw.annotations:
        if an['description'].find('bad_segment_eeg_raw') >= 0:
            bads += an['duration']
        elif an['description'].find('bad_segment_eeg_diff') >= 0:
            dbads += an['duration']

    # From ica._get_infos_for_repr
    abs_vars = ica.pca_explained_variance_
    rel_vars = abs_vars / abs_vars.sum()
    fit_explained_variance = rel_vars[:ica.n_components_].sum()

    info = {'subj_id': subj_id,
            'total_ica': len(np.unique(ica.exclude)),
            'VEOG': len(ica.labels_['eog/0/VEOG']),
            'HEOG': len(ica.labels_['eog/0/HEOG']),
            'PCA_explained': fit_explained_variance,
            'bad_seg_duration': bads,
            'diff_bad_seg_duration': dbads}
    preprocs.append(info)

#%% --------------------------------------------------------------
# Print a summary

df = pd.DataFrame(preprocs)
print(df.describe())
