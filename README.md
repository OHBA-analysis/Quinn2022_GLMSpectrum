# The GLM-Spectrum: Multilevel power spectrum analysis with covariate and confound modelling.
This repository contains the scripts and software to run the simulations and real data analysis published in:

Andrew J. Quinn, Lauren Atkinson, Chetan Gohil, Oliver Kohl, Jemma Pitt, Catharina Zich, Anna C. Nobre & Mark W. Woolrich (2022) The GLM-Spectrum: Multilevel power spectrum analysis with covariate and confound modelling

## Getting started

First, clone this repository into a directory on your computer:

```
git clone https://github.com/OHBA-analysis/Quinn2022_GLMSpectrum.git
```

then create a conda environment and install the dependencies

```
conda env create -f glmspectrum_env.yml
conda activate glm-spectrum
```

Next, you need to configure the `lemon_raw` and `lemon_output` directories in `glm_config.yml`. `lemon_raw` specifies a directory where the raw data will be downloaded to (or where the raw data already exists) and `lemon_output` specifies a directory where the generatedoutputs from this analysis will be stored.

After specifying these paths, `glm_config.yml` should look something like this:

```
lemon_raw: /path/to/my/raw/data_folder
lemon_output: /path/to/my/output_folder
lemon_raw_url: https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/
lemon_behav_url: https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/Behavioural_Data_MPILMBB_LEMON/
```

From here you can run the analysis, plotting and supplemental scripts in order. Outputs will be saved into your `lemon_output` directory.


## Requirements

A full list of requirements is specified in the `requirements.txt` file and the `glmspectrum_env.yml` anaconda environment.

The EEG data analysis depends on [MNE-Python](https://mne.tools/stable/index.html) and [OSL](https://github.com/OHBA-analysis/osl). The GLM-Spectrum used in this paper is implemented in the [SAILS toolbox](https://joss.theoj.org/papers/10.21105/joss.01982) as `sails.stft.glm_periodogram`. Another implementation is available in [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics) as `osl_dynamics.analysis.regression_spectra`. The GLM analysis and statistics further depend on [glmtools](https://pypi.org/project/glmtools/)

## Data

This paper uses the open-data availiable from the mind-body-brain dataset.

Babayan, A., Erbey, M., Kumral, D. et al. A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. Sci Data 6, 180308 (2019). 
https://doi.org/10.1038/sdata.2018.308
