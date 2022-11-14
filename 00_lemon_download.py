"""This script will download the raw EEG data from the MPI-Leipzig_Mind-Brain-Body-LEMON dataset.

This can be run once before the start of any analyses. It can be skipped if the
data are already present on disk in BIDS format.

"""

import os
import urllib.request
from pathlib import Path

import bs4
import requests

from glm_config import cfg

#%% ------------------------------------------------------------
# Download raw EEG data

datadir = cfg['lemon_raw_eeg']

url = cfg['lemon_raw_url']
r = requests.get(url)
data = bs4.BeautifulSoup(r.text, "html.parser")

for l in data.find_all("a"):
    subj_id = l.get_text()[:-1]

    subj_url = url + l.get_text() + '/RSEEG'

    subj_r = requests.get(subj_url)
    subj_data = bs4.BeautifulSoup(subj_r.text, "html.parser")

    if len(subj_data.find_all("a")) == 4:
        local_dir = os.path.join(datadir, subj_id, 'RSEEG')
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        for m in subj_data.find_all("a"):
            if m.get_text() == '../':
                continue
            remote_file = subj_url + '/' + m.get_text()
            local_file = os.path.join(local_dir, m.get_text())
            print(local_file)

            urllib.request.urlretrieve(remote_file, filename=local_file)


#%% ---------------------------
#
# header errors (wrong DataFile and MarkerFile) in:
# sub-010193.vhdr
# sub-010219.vhdr
#
# Can be fixed by hand


#%% ------------------------------------------------------------
# Download Subject metadata


datadir = cfg['lemon_raw']
fname = 'META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'

url = cfg['lemon_behav_url'] + fname

local_file = os.path.join(os.path.dirname(cfg['lemon_raw'].rstrip('/')), fname)
urllib.request.urlretrieve(url, filename=local_file)
