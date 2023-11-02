"""This script produces a figure axis scaling options."""

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import osl
import sails

import lemon_plotting
from glm_config import cfg
from lemon_support import get_eeg_data

#%% ----------------------------------------------------------
# Prep

fbase = os.path.join(cfg['lemon_processed_data'], '{subj}/{subj}_preproc_raw.fif')
st = osl.utils.Study(fbase)

fname = st.get(subj='sub-010060')[0]

runname = fname.split('/')[-1].split('.')[0]
print('processing : {0}'.format(runname))

subj_id = osl.utils.find_run_id(fname)

raw = mne.io.read_raw_fif(fname, preload=True)
refraw = raw.copy().pick_types(eeg=True)

# Get data
XX = get_eeg_data(raw).T
fs = raw.info['sfreq']

# Simple model
f, pxx = sails.stft.periodogram(XX, axis=0,
                                nperseg=int(fs*2),
                                noverlap=int(fs),
                                fmin=0.1, fmax=100,
                                fs=fs, mode='magnitude')

#%% ------------------------------------------------------------


# https://docs.python.org/2/tutorial/modules.html
def fib2(n):   # return Fibonacci series up to n
    """Compute fibonacci sequence."""
    result = []
    a, b = 0, 1
    while b < n:
        result.append(b)
        a, b = b, a+b
    return result


fx = fib2(200)[1:]

# Map freq to indices of fibonacci
f_fib = np.zeros_like(f)
xtks = []
for ii in range(len(f_fib)):
    if f[ii] <= 1:
        # Stay linear below 1
        f_fib[ii] = f[ii]
    else:
        # find fib in just below or equivalent to f
        ind1 = np.argmin(np.abs(f[ii] - fx))
        ind2 = (f[ii] - fx[ind1]) / (fx[ind1+1] - fx[ind1])
        f_fib[ii] = ind1 + ind2 + 1
    if f[ii] in fx:
        xtks.append(ii)

#%% ----------------------------------------------------------
# Plot

plt.figure(figsize=(12, 6.75))
plt.subplots_adjust(bottom=0.25, top=0.75, wspace=0.3, left=0.05, right=0.95)

ax = plt.subplot(1, 4, 1)
lemon_plotting.plot_sensor_spectrum(ax, pxx, refraw, f, base=1)
ax.set_title('linear-scale')

ax = plt.subplot(1, 4, 2)
lemon_plotting.plot_sensor_spectrum(ax, np.sqrt(pxx), refraw, f, base=0.5)
ax.set_title('sqrtsqrt-scale')

ax = plt.subplot(1, 4, 3)
lemon_plotting.plot_sensor_spectrum(ax, np.sqrt(pxx), refraw, f_fib, base=1)
plt.xticks(f_fib[xtks], fx[:-1])
ax.set_title('fibonacci-sqrt-scale')

ax = plt.subplot(1, 4, 4)
lemon_plotting.plot_sensor_spectrum(ax, pxx, refraw, f, base=1)
ax.set_title('loglog-scale')
ax.set_yscale('log')
ax.set_xscale('log')

fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_lin-sqrt-log.png')
plt.savefig(fout, transparent=True, dpi=300)
