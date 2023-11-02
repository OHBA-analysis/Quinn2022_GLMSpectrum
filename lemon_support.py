
import os
import osl
import numpy as np
import mne
import sails
from scipy import io, ndimage, stats
import matplotlib.pyplot as plt

from glm_config import cfg

import logging
logger = logging.getLogger('osl')


def lemon_set_standard_montage(glmsp):
    from pathlib import Path
    pth = Path(mne.channels.__file__).parent
    pth = pth / 'data' / 'montages' / 'brainproducts-RNP-BA-128.txt'

    fbase = os.path.join(cfg['lemon_processed_data'], 'sub-010002', 'sub-010002_preproc_raw.fif')
    reference = mne.io.read_raw_fif(fbase).pick_types(eeg=True)

    new = []
    with open(pth, 'r') as f:
        for line in f.readlines():
            tag = line.split(' ')[0]
            if tag == 'Name' or tag in reference.ch_names:
                new.append(line)
    new_mon = os.path.join(cfg['code_dir'], 'lemon_custom_montage.txt')
    with open(new_mon, 'w') as f:
        for line in new:
            f.write(line)

    from mne.channels._standard_montage_utils import _read_theta_phi_in_degrees
    mon =  _read_theta_phi_in_degrees(new_mon, mne.defaults.HEAD_SIZE_DEFAULT)
    reference = reference.set_montage(mon)
    glmsp.info = reference.info
    return glmsp


def lemon_set_channel_montage(dataset, userargs):
    logger.info('LEMON Stage - load and set channel montage')
    logger.info('userargs: {0}'.format(str(userargs)))

    subj = '010060'
    #base = f'/Users/andrew/Projects/lemon/EEG_Raw_BIDS_ID/sub-{subj}/RSEEG/'
    #base = f'/ohba/pi/knobre/datasets/MBB-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/sub-{subj}/RSEEG/'
    ref_file = os.path.join(cfg['lemon_raw'], f'sub-{subj}', 'RSEEG', f'sub-{subj}.mat')
    ref_file = os.path.join(cfg['code_dir'], 'sub-010060.mat')
    X = io.loadmat(ref_file)
    ch_pos = {}
    for ii in range(len(X['Channel'][0])-1):  #final channel is reference
        key = X['Channel'][0][ii][0][0].split('_')[2]
        if key[:2] == 'FP':
            key = 'Fp' + key[2]
        value = X['Channel'][0][ii][3][:, 0]
        value = np.array([value[1], value[0], value[2]])
        ch_pos[key] = value

    dig = mne.channels.make_dig_montage(ch_pos=ch_pos)
    dataset['raw'].set_montage(dig)

    return dataset


def get_eeg_data(raw, csd=True):
    """Load EEG and perform sanity checks."""

    # Use first scan as reference for channel labels and order
    fbase = os.path.join(cfg['lemon_processed_data'], 'sub-010002', 'sub-010002_preproc_raw.fif')
    reference = mne.io.read_raw_fif(fbase).pick_types(eeg=True)
    mon = reference.get_montage()

    # Load ideal layout and match data-channels
    raw = raw.copy().pick_types(eeg=True)
    ideal_inds = [mon.ch_names.index(c) for c in raw.info['ch_names']]

    if csd:
        # Apply laplacian if requested
        raw = mne.preprocessing.compute_current_source_density(raw)
        X = raw.get_data(picks='csd')
    else:
        # Get data from EEG picks
        X = raw.get_data(picks='eeg')

    # Preallocate & store ouput
    Y = np.zeros((len(mon.ch_names), X.shape[1]))

    Y[ideal_inds, :] = X

    return Y


def lemon_create_heog(dataset, userargs):
    logger.info('LEMON Stage - Create HEOG from F7 and F8')
    logger.info('userargs: {0}'.format(str(userargs)))

    F7 = dataset['raw'].get_data(picks='F7')
    F8 = dataset['raw'].get_data(picks='F8')

    heog = F7-F8

    info = mne.create_info(['HEOG'],
                           dataset['raw'].info['sfreq'],
                           ['eog'])
    eog_raw = mne.io.RawArray(heog, info)
    dataset['raw'].add_channels([eog_raw], force_update_info=True)

    return dataset


def lemon_ica(dataset, userargs, logfile=None):
    logger.info('LEMON Stage - custom EEG ICA function')
    logger.info('userargs: {0}'.format(str(userargs)))

    # NOTE: **userargs doesn't work because 'picks' is in there
    ica = mne.preprocessing.ICA(n_components=userargs['n_components'],
                                max_iter=1000,
                                random_state=42)

    # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#filtering-to-remove-slow-drifts
    fraw = dataset['raw'].copy().filter(l_freq=1., h_freq=None)

    ica.fit(fraw, picks=userargs['picks'])
    dataset['ica'] = ica

    logger.info('starting EOG autoreject')
    # Find and exclude VEOG
    #eog_indices, eog_scores = dataset['ica'].find_bads_eog(dataset['raw'])
    veog_indices, eog_scores =  dataset['ica'].find_bads_eog(dataset['raw'], 'VEOG')
    if len(veog_indices) == 0:
        veog_indices, eog_scores =  dataset['ica'].find_bads_eog(dataset['raw'], 'VEOG', threshold=2)
    dataset['veog_scores'] = eog_scores
    dataset['ica'].exclude.extend(veog_indices)
    logger.info('Marking {0} ICs as EOG {1}'.format(len(dataset['ica'].exclude),
                                                    veog_indices))

    # Find and exclude HEOG
    #heog_indices = lemon_find_heog(fraw, ica)
    heog_indices, eog_scores =  dataset['ica'].find_bads_eog(dataset['raw'], 'HEOG')
    if len(heog_indices) == 0:
        heog_indices, eog_scores =  dataset['ica'].find_bads_eog(dataset['raw'], 'HEOG', threshold=2)
    dataset['heog_scores'] = eog_scores
    dataset['ica'].exclude.extend(heog_indices)
    logger.info('Marking {0} ICs as HEOG {1}'.format(len(heog_indices),
                                                     heog_indices))

   # Save components as channels in raw object
    src = dataset['ica'].get_sources(fraw).get_data()
    veog = src[veog_indices[0], :]
    heog = src[heog_indices[0], :]

    ica.labels_['top'] = [veog_indices[0], heog_indices[0]]

    info = mne.create_info(['ICA-VEOG', 'ICA-HEOG'],
                           dataset['raw'].info['sfreq'],
                           ['misc', 'misc'])
    eog_raw = mne.io.RawArray(np.c_[veog, heog].T, info)
    dataset['raw'].add_channels([eog_raw], force_update_info=True)

    # Apply ICA denoising or not
    if ('apply' not in userargs) or (userargs['apply'] is True):
        logger.info('Removing selected components from raw data')
        dataset['ica'].apply(dataset['raw'])
    else:
        logger.info('Components were not removed from raw data')
    return dataset


def lemon_zapline_dss(dataset, userargs, logfile=None):
    logger.info('LEMON Stage - ZapLine power removal')
    logger.info('userargs: {0}'.format(str(userargs)))
    from meegkit import dss
    # https://mne.discourse.group/t/clean-line-noise-zapline-method-function-for-mne-using-meegkit-toolbox/7407
    fline = userargs.get('fline', 50)

    data = dataset['raw'].get_data() # Convert mne data to numpy darray
    sfreq = dataset['raw'].info['sfreq'] # Extract the sampling freq
   
    #Apply MEEGkit toolbox function
    out, _ = dss.dss_line(data.T, fline, sfreq, nremove=4) # fline (Line noise freq) = 50 Hz for Europe

    dataset['raw']._data = out.T # Overwrite old data

    return dataset


def lemon_make_task_regressor(dataset):
    ev, ev_id = mne.events_from_annotations(dataset['raw'])
    print('Found {0} events in raw'.format(ev.shape[0]))
    print(ev_id)

    # Correct for cropping first 10 seconds - not sure why this is necessary?!
    ev[:, 0] -= dataset['raw'].first_samp

    task = np.zeros((dataset['raw'].n_times,))
    for ii in range(ev.shape[0]):
        if ev[ii, 2] == ev_id['Stimulus/S200']:
            # EYES OPEN
            task[ev[ii,0]:ev[ii,0]+5000] = 1
        elif ev[ii, 2] == ev_id['Stimulus/S210']:
            # EYES CLOSED
            task[ev[ii,0]:ev[ii,0]+5000] = -1
        elif ev[ii, 2] == 1:
            task[ev[ii,0]] = task[ev[ii,0]-1]

    return task


def find_eog_events(raw, event_id=998):
    eog = raw.copy().filter(l_freq=1, h_freq=10, picks='eog').get_data(picks='VEOG')
    eog = eog[0, :]
    # 10 seconds hopefully long enough to avoid rejecting real blinks - only
    # want to catch HUGE artefacts here.
    bads = sails.utils.detect_artefacts(eog, axis=0, reject_mode='segments', segment_len=2500)
    eog[bads] = np.median(eog)
    logger.info('Removed {0} bad samples from EOG ({1}%)'.format(bads.sum(), 100*(bads.sum()/len(bads))))

    if np.abs(np.max(eog)) > np.abs(np.min(eog)):
        eog_events, _ = mne.preprocessing.eog.peak_finder(eog,
                                                          None, extrema=1)
    else:
        eog_events, _ = mne.preprocessing.eog.peak_finder(eog,
                                                          None, extrema=-1)

    n_events = len(eog_events)
    logger.info(f'Number of EOG events detected: {n_events}')
    #eog_events = np.array([eog_events + raw.first_samp,
    eog_events = np.array([eog_events,
                           np.zeros(n_events, int),
                           event_id * np.ones(n_events, int)]).T

    return eog_events


def lemon_make_blinks_regressor(raw, corr_thresh=0.75, figpath=None):
    #eog_events = mne.preprocessing.find_eog_events(raw, l_freq=1, h_freq=10)
    eog_events = find_eog_events(raw)
    logger.info('found {0} blinks'.format(eog_events.shape[0]))
    #eog_events = find_eog_events(raw)
    #logger.info('found {0} blinks'.format(eog_events.shape[0]))

    # Correct for cropping first 10 seconds - not sure why this is necessary?!
    #eog_events[:, 0] -= int(10*raw.info['sfreq'])

    tmin = -0.1
    tmax = 0.15
    epochs = mne.Epochs(raw, eog_events, 998, tmin, tmax, picks='eog')
    ev_eog = epochs.get_data()[:, 0, :]
    C = np.abs(np.corrcoef(ev_eog.mean(axis=0), ev_eog)[1:,0])
    drops = np.where(C < corr_thresh)[0]
    clean = epochs.copy().drop(drops)
    keeps = np.where(C > corr_thresh)[0]
    dirty = epochs.copy().drop(keeps)

    eog_events = np.delete(eog_events, drops, axis=0)
    logger.info('found {0} clean blinks'.format(eog_events.shape[0]))

    blink_covariate = np.zeros((raw.n_times,))
    #blink_covariate[eog_events[:, 0] - raw.first_samp] = 1
    blink_covariate[eog_events[:, 0]] = 1
    blink_covariate = ndimage.maximum_filter(blink_covariate,
                                             size=raw.info['sfreq']//2)

    if figpath is not None:
        plt.figure(figsize=(16, 10))
        plt.subplot(231)
        plt.plot(epochs.times, epochs.get_data()[:, 0, :].mean(axis=0))
        plt.title('All blinks')
        plt.subplot(234)
        plt.plot(epochs.times, epochs.get_data()[:, 0, :].T)
        plt.subplot(232)
        plt.plot(epochs.times, clean.get_data()[:, 0, :].mean(axis=0))
        plt.title('Clean blinks')
        plt.subplot(235)
        plt.plot(epochs.times, clean.get_data()[:, 0, :].T)
        plt.subplot(233)
        plt.title('Dirty blinks')
        plt.plot(epochs.times, dirty.get_data()[:, 0, :].mean(axis=0))
        plt.subplot(236)
        plt.plot(epochs.times, dirty.get_data()[:, 0, :].T)
        plt.savefig(figpath, transparent=False, dpi=300)

    return blink_covariate, eog_events.shape[0], clean.average(picks='eog')


def lemon_make_bads_regressor(raw, mode='eeg'):
    bads = np.zeros((raw.n_times,))
    for an in raw.annotations:
        if an['description'].startswith('bad') and an['description'].endswith(mode):
            start = raw.time_as_index(an['onset'])[0] - raw.first_samp
            duration = int(an['duration'] * raw.info['sfreq'])
            bads[start:start+duration] = 1
    if mode == 'raw':
        bads[:int(raw.info['sfreq']*2)] = 1
        bads[-int(raw.info['sfreq']*2):] = 1
    else:
        bads[:int(raw.info['sfreq'])] = 1
        bads[-int(raw.info['sfreq']):] = 1
    return bads


def quick_plot_eog_icas(raw, ica, figpath=None):

    inds = np.arange(250*45, 250*300)

    plt.figure(figsize=(16, 9))
    veog = raw.get_data(picks='VEOG')[0, :]
    ica_veog = raw.get_data(picks='ICA-VEOG')[0, :]
    ax = plt.axes([0.05, 0.55, 0.125, 0.4])
    comp = ica.get_components()[:, ica.labels_['top'][0]]
    mne.viz.plot_topomap(comp, ica.info, axes=ax, show=False)

    plt.axes([0.2, 0.55, 0.475, 0.4])
    plt.plot(stats.zscore(veog[inds]))
    plt.plot(stats.zscore(ica_veog[inds])-10)
    plt.legend(['VEOGs', 'ICA-VEOG'], frameon=False)
    plt.xlim(0, 250*180)

    plt.axes([0.725, 0.55, 0.25, 0.4])
    plt.plot(veog, ica_veog, '.k')
    veog = raw.get_data(picks='VEOG', reject_by_annotation='omit')[0, :]
    ica_veog = raw.get_data(picks='ICA-VEOG', reject_by_annotation='omit')[0, :]
    plt.plot(veog, ica_veog, '.r')
    plt.xlabel('VEOG'); plt.ylabel('ICA-VEOG')
    plt.plot(veog, ica_veog, '.r')
    plt.legend(['Samples', 'Clean Samples'], frameon=False)
    plt.title('Correlation : r = {0}'.format(np.corrcoef(veog, ica_veog)[0,  1]))

    heog = raw.get_data(picks='HEOG')[0, :]
    ica_heog = raw.get_data(picks='ICA-HEOG')[0, :]
    plt.axes([0.05, 0.05, 0.125, 0.4])
    comp = ica.get_components()[:, ica.labels_['top'][1]]
    mne.viz.plot_topomap(comp, ica.info, show=False)

    plt.axes([0.2, 0.05, 0.475, 0.4])
    plt.plot(stats.zscore(heog[inds]))
    plt.plot(stats.zscore(ica_heog[inds])-5)
    plt.legend(['HEOGs', 'ICA-HEOG'], frameon=False)
    plt.xlim(0, 250*180)

    plt.axes([0.725, 0.05, 0.25, 0.4])
    plt.plot(heog, ica_heog, '.k')
    heog = raw.get_data(picks='HEOG', reject_by_annotation='omit')[0, :]
    ica_heog = raw.get_data(picks='ICA-HEOG', reject_by_annotation='omit')[0, :]
    plt.plot(heog, ica_heog, '.r')
    plt.legend(['Samples', 'Clean Samples'], frameon=False)
    plt.xlabel('HEOG'); plt.ylabel('ICA-HEOG')
    plt.title('Correlation : r = {0}'.format(np.corrcoef(heog, ica_heog)[0,  1]))

    plt.savefig(figpath, transparent=False, dpi=300)


def quick_plot_eog_epochs(raw, figpath=None):

    fig = mne.preprocessing.create_eog_epochs(raw, picks='eeg').average().plot_joint(show=False)
    fig.savefig(figpath.format('eeg_eog_epochs'))

    fig = mne.preprocessing.create_eog_epochs(raw).average().plot(show=False)
    fig.savefig(figpath.format('eog_eog_epochs'))


def plot_design(ax, design_matrix, regressor_names):
    num_observations, num_regressors = design_matrix.shape
    vm = np.max((design_matrix.min(), design_matrix.max()))
    cax = ax.pcolor(design_matrix, cmap=plt.cm.coolwarm,
                    vmin=-vm, vmax=vm)
    ax.set_xlabel('Regressors')
    tks = np.arange(len(regressor_names)+1)
    ax.set_xticks(tks+0.5)
    ax.set_xticklabels(tks)

    tkstep = 2
    tks = np.arange(0, design_matrix.shape[0], tkstep)

    for tag in ['top', 'right', 'left', 'bottom']:
        ax.spines[tag].set_visible(False)

    summary_lines = True
    new_cols = 0
    for ii in range(num_regressors):
        if summary_lines:
            x = design_matrix[:, ii]
            if np.abs(np.diff(x)).sum() != 0:
                y = (0.5*x) / (np.max(np.abs(x)) * 1.1)
            else:
                # Constant regressor
                y = np.ones_like(x) * .45
            if num_observations > 50:
                ax.plot(y+ii+new_cols+0.5, np.arange(0, 0+num_observations)+.5, 'k')
            else:
                yy = y+ii+new_cols+0.5
                print('{} - {} - {}'.format(yy.min(), yy.mean(), yy.max()))
                ax.plot(y+ii+new_cols+0.5, np.arange(0, 0+num_observations)+.5,
                        'k|', markersize=5)

        # Add white dividing line
        if ii < num_regressors-1:
            ax.plot([ii+1+new_cols, ii+1+new_cols], [0, 0+num_observations],
                    'w', linewidth=4)
    return cax
