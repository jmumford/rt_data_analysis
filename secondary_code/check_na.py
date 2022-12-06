import glob
from pathlib import Path

def get_subids(root):
    subdirs = glob.glob(f'{root}/s*/')
    subid = [val[-4:-1] for val in subdirs]
    return subid

root = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS_scans'

tasks = ['stroop', 'ANT',  'stopSignal', 'twoByTwo',
                 'discountFix', 'DPX', 'motorSelectiveStop','CCTHot', 'WATT3']


subs = get_subids(root)

subid = []
task_vec = []
total_key_press_1 =[]
total_missing_rt = []
total_both_key_press_and_missing = []
for task in tasks:
    for sub in subs:
        file_path = glob.glob(
            f'{root}/sub-s{sub}/ses-[0-9]/func/*{task}*tsv'
        )
        if len(file_path) == 1:
            events_df = pd.read_csv(file_path[0], sep = '\t')
            subid.append(sub)
            task_vec.append(task)
            total_key_press_1.append(np.sum(events_df.key_press == -1))
            total_missing_rt.append(np.sum(events_df.response_time.isnull()))
            total_both_key_press_and_missing.append(np.sum((events_df.key_press == -1) & (events_df.response_time.isnull())))


np.sum(total_both_key_press_and_missing != total_missing_rt)
np.sum(total_both_key_press_and_missing != total_total_key_press_1)


max_planning_dur = []
min_num_zeros = []
max_num_zeros = []
subid = []
for sub in subs:
    file_path = glob.glob(
            f'{root}/sub-s{sub}/ses-[0-9]/func/*WATT3*tsv'
        )
    if len(file_path) == 1:
        events_df = pd.read_csv(file_path[0], sep = '\t')
        max_planning_dur.append(events_df.query('planning == 1')[['block_duration']].block_duration.max()/1000)
        subid.append(sub)

        a = events_df['planning'].values
        m1 = np.r_[False, a==0, False]
        idx = np.flatnonzero(m1[:-1] != m1[1:])
        out = (idx[1::2]-idx[::2])
        min_num_zeros.append(out.min())
        max_num_zeros.append(out.max())

print(np.max(max_planning_dur))
print(np.max(max_num_zeros))
print(np.min(min_num_zeros))