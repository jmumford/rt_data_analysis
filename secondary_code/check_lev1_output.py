import glob
import pandas as pd
import numpy as np


def get_subids(root):
    subdirs = glob.glob(f'{root}/s*/')
    subid = [val[-4:-1] for val in subdirs]
    return subid

root = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS_scans'

path = '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output'

dirs = glob.glob(f'{path}/*lev1*/task*/')

num_good = []
num_con_est = []
diff_btwn = []
for dir in dirs:
    csv_file = glob.glob(f'{dir}/*csv')[0]
    csv_data = pd.read_csv(csv_file)
    csv_subids = set(csv_data.subid_task.str.split('_', expand=True)[0])
    num_good.append(110 - len(csv_subids))
    con_files = glob.glob(f'{dir}/contrast_estimates/*.nii.gz')
    subids = set(np.unique([val.split('sub_')[1][:3] for val in con_files]))
    num_con_est.append(len(subids))
    diff_btwn.append(len(subids) - (110 - len(csv_subids)))

diff_array = np.array(diff_btwn)
dirs_array = np.array(dirs)
dirs_array[diff_array != 0]


all_subs = set(get_subids(root))
dir = dirs_array[diff_array != 0][0]
csv_file = glob.glob(f'{dir}/*csv')[0]
csv_data = pd.read_csv(csv_file)
csv_subids = set(csv_data.subid_task.str.split('_', expand=True)[0])
con_files = glob.glob(f'{dir}/contrast_estimates/*.nii.gz')
subids_con_files = set(np.unique([val.split('sub_')[1][:3] for val in con_files]))

all_subs - csv_subids - subids_con_files