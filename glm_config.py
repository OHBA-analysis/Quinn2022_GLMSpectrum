import os
import yaml
import pathlib
import warnings

code_dir = str(pathlib.Path(__file__).parent.resolve())

yaml_path = os.path.join(code_dir, 'glm_config.yaml')
with open(yaml_path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg['yaml_path'] = yaml_path
cfg['code_dir'] = code_dir

# Check LEMON
cfg['lemon_raw_eeg'] = os.path.join(cfg['lemon_raw'], 'EEG_Raw_BIDS_ID')
cfg['lemon_processed_data'] = os.path.join(cfg['lemon_output'], 'preprocessed_data')
cfg['lemon_glm_data'] = os.path.join(cfg['lemon_output'], 'glm_data')
cfg['lemon_figures'] = os.path.join(cfg['lemon_output'], 'figures')

# Check that these folders exist and warn if they don't
for directory in ['lemon_raw', 'lemon_raw_eeg', 'yaml_path', 'code_dir']:
    if not os.path.exists(cfg[directory]):
        warnings.warn('WARNING: dir not found - {0} : {1}'.format(directory, cfg[directory]))

# Check that these folders exist and make them if they don't
for directory in ['lemon_output', 'lemon_processed_data', 'lemon_glm_data', 'lemon_glm_data']:
    if not os.path.exists(cfg[directory]):
        os.makedirs(cfg[directory])
