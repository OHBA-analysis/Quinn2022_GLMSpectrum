"""This script creates figure 1."""

import os

import glmtools as glm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import lemon_plotting
from glm_config import cfg

outdir = cfg['lemon_figures']

#%% ------------------------------------------------
# Data scatter example

const = np.ones((128,))
bads = np.zeros((128,))
bads[35:40] = 1
bads[99:108] = 1

Y = np.random.randn(128,) + 1
Y[bads.astype(bool)] += 4
data = glm.data.TrialGLMData(data=Y)

X1 = np.vstack((const,)).T
regressor_names1 = ['Simple Mean']
C1 = np.eye(1)
contrast_names1 = ['Simple Mean']

design1 = glm.design.GLMDesign.initialise_from_matrices(X1, C1,
                                                        regressor_names=regressor_names1,
                                                        contrast_names=contrast_names1)

X2 = np.vstack((const, bads)).T
regressor_names2 = ['Intercept', 'Artefact']
C2 = np.eye(2)
C2 = np.array([[1, 0], [0, 1], [1, 1]])
contrast_names2 = ['Intercept', 'Artefact Effect', 'Artefact Mean']

design2 = glm.design.GLMDesign.initialise_from_matrices(X2, C2,
                                                        regressor_names=regressor_names2,
                                                        contrast_names=contrast_names2)
model = glm.fit.OLSModel(design2, data)


fig = plt.figure(figsize=(16, 9))

# Plot first design
ax = plt.subplot(1, 4, 1)
glm.viz.plot_design_summary(X1, regressor_names1,
                            contrasts=C1, contrast_names=contrast_names1,
                            ax=ax)
pos1 = ax.get_position()  # get the original position
pos2 = [pos1.x0 - 0.035, pos1.y0+0.07,  pos1.width, pos1.height*0.92]
ax.set_position(pos2)  # set a new position
lemon_plotting.subpanel_label(ax, 'A')

# Plot second design
ax = plt.subplot(1, 4, 2)
glm.viz.plot_design_summary(X2, regressor_names2,
                            contrasts=C2, contrast_names=contrast_names2,
                            ax=ax)
lemon_plotting.subpanel_label(ax, 'B')

ax = plt.axes((0.62, 0.2, 0.35, 0.6))
# Calculate the point density
xy = np.vstack([bads, Y])
z = gaussian_kde(xy)(xy)

# Plot line for simple mean term
plt.scatter(bads, Y, c=z, s=100)
plt.plot((-0.1, 1.1), (Y.mean(), Y.mean()), 'k--')
plt.text(0.75, 1.55, 'Simple Mean')

# Plot line for intercept
plt.plot((-0.1, 0.1), (model.betas[0, 0], model.betas[0, 0]), 'k--')
plt.text(0.055, 0.25, 'Intercept\n(Conditioned Mean)')

# Plot line for artefact mean term
plt.plot((0.9, 1.1), (model.copes[2, 0], model.copes[2, 0]), 'k--')
plt.text(1.055, 5, 'Artefact Mean')

# Plot line for regressor effect
plt.plot(np.linspace(0, 1),  model.betas[0, 0] + np.linspace(0, 1) * model.betas[1, 0])
plt.text(0.35, 3, 'Artefact\nEffect')
plt.xlabel('Artefact Regressor Value')
plt.ylabel('Observed Data')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)

# colourbar to indicate point density
plt.colorbar(label='Point Density', shrink=0.5)
lemon_plotting.subpanel_label(ax, 'C')

fig.axes[1].set_visible(False)
fig.axes[3].set_position([0.51, 0.4, 0.01, 0.35])  # set a new position

fout = os.path.join(cfg['lemon_figures'], 'glm-spectrum_example-designs-confound.png')
plt.savefig(fout, dpi=300, transparent=True)
fout = os.path.join(cfg['lemon_figures'], 'glm-spectrum_example-designs-confound_low-res.png')
plt.savefig(fout, dpi=100, transparent=True)