"""This script runs the preprocessing of the raw EEG data.

The preprocessing chain is defined in lemon_preproc.yml - you can adapt the
preprocessing by changing the values in this file.

The script can be set to preprocess just the single example subject by changing
'subj' in the main body.

"""

import os

import osl
from dask.distributed import Client

from glm_config import cfg
from lemon_support import (lemon_create_heog, lemon_ica,
                           lemon_set_channel_montage)

#%% ------------------------------
# User options to set

subj = 'all'  # Set to 'sub-010060' to run single subject in paper examples
n_dask_workers = 8  # Number of dask parallel workers to use if subj is 'all'

#%% -------------------------------
# Main code

if __name__ == '__main__':

    extra_funcs = [lemon_set_channel_montage, lemon_create_heog, lemon_ica]

    fbase = os.path.join(cfg['lemon_raw_eeg'], '{subj}', 'RSEEG', '{subj}.vhdr')
    st = osl.utils.Study(fbase)

    config = osl.preprocessing.load_config('lemon_preproc.yml')
    proc_outdir = cfg['lemon_processed_data']

    if subj == 'all':
        # Run everybody using a dask cluster
        client = Client(n_workers=n_dask_workers, threads_per_worker=1)
        goods = osl.preprocessing.run_proc_batch(config, st.match_files, proc_outdir,
                                                 overwrite=True,
                                                 extra_funcs=extra_funcs,
                                                 dask_client=True)
    else:
        # Run a single subject
        inputs = st.get(subj=subj)
        if len(inputs) == 0:
            print(f"Subject '{subj}' not found in raw_data directory")
        osl.preprocessing.run_proc_chain(inputs[0], config,
                                         outdir=proc_outdir,
                                         extra_funcs=extra_funcs)
