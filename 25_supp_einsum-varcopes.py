import numpy as np
import timeit
import os
import matplotlib.pyplot as plt
from glm_config import cfg

# https://stackoverflow.com/a/72875625
import numba as nb
@nb.njit(parallel=True)  # , fastmath=True
def diag_dot(a, b):
    res = np.zeros(a.shape[0])
    for i in nb.prange(a.shape[0]):
        for j in range(a.shape[1]):
            res[i] += a[i, j] * b[j, i]
    return res
# run once to get compile out of the way
xx = diag_dot(np.arange(9).reshape(3,3), np.arange(9).reshape(3,3))

#Your statements here
nruns = 100
nobservations = 25
ntests = [1, 100, 100 * 61, 100 * 204]

t = np.zeros((4, 4, nruns))

# Takes a few minutes to compute   
msg = 'Running dataset {0}/4 of shape {1} - each gets exponentially slower...'
for jj in range(4):
    print(msg.format(jj+1, (nobservations, ntests[jj])))
    for ii in range(t.shape[2]):

        # test data - content doesn't matter
        resid = np.random.randn(nobservations, ntests[jj])

        # Run dot-diag method
        start = timeit.default_timer()
        resid_dots1 = np.diag( resid.T.dot(resid) )
        stop = timeit.default_timer()
        t[jj, 0, ii] = stop - start

        # Run einsum method
        start = timeit.default_timer()
        resid_dots2 = np.einsum('ij,ji->i', resid.T, resid)
        stop = timeit.default_timer()
        t[jj, 1, ii] = stop - start

        # Run numba parallel method
        start = timeit.default_timer()
        resid_dots3 = diag_dot(resid.T, resid)
        stop = timeit.default_timer()
        t[jj, 2, ii] = stop - start

        # Run sensible normal method
        start = timeit.default_timer()
        resid_dots4 = (resid**2).sum(axis=0)
        stop = timeit.default_timer()
        t[jj, 3, ii] = stop - start

        assert(np.allclose(resid_dots1, resid_dots2))
        assert(np.allclose(resid_dots1, resid_dots3))

plt.figure(figsize=(16,6))
plt.axes([0.12, 0.1, 0.7, 0.7])
xpos = np.arange(20).reshape(4, 5)[:, :4].reshape(-1)
h = plt.boxplot(t[:, :, :].reshape(16, nruns).T, positions=xpos)
plt.text(1.5, 7, '25 Windows\nSingle Channel\nSingle Frequency', ha='center')
plt.text(6.5, 7, '25 Windows\nSingle Channel\n100 Frequency Bins', ha='center')
plt.text(11.5, 7, '25 Windows\n60 Channels\n100 Frequency Bins', ha='center')
plt.text(16.5, 7, '25 Windows\n204 Channel\n100 Frequency Bins', ha='center')
plt.yscale('log')
plt.yticks([10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1],
           ['1/100 millisecond', '1/10 millisecond', '1 millisecond', '10 milliseconds', '100 milliseconds', '1 Second'])
plt.xticks(xpos, ['DotDiag', 'EinSum', 'Numba', 'SumSqr'] * 4)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('VARCOPE Method')
plt.ylabel('Time for single permutation')
plt.grid(axis='y', lw=0.2)
fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_einsum-varcopes.png')
plt.savefig(fout, transparent=True, dpi=300)
fout = os.path.join(cfg['lemon_figures'], 'lemon-supp_einsum-varcopes_low-res.png')
plt.savefig(fout, transparent=True, dpi=100)

