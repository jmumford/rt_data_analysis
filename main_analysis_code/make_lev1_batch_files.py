#!/usr/bin/env python

import glob
from pathlib import Path

def get_subids(root):
    subdirs = glob.glob(f'{root}/s*/')
    subid = [val[-4:-1] for val in subdirs]
    return subid


tasks = ['stroop', 'ANT',  'stopSignal', 'twoByTwo',
                 'discountFix', 'DPX', 'motorSelectiveStop','CCTHot', 'WATT3']

#tasks = ['twoByTwo']

batch_stub = ('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/'
              'rt_data_analysis/main_analysis_code/run_stub.batch')
root = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS_scans'


# For Jeanette's study no_rt is studied.  For other studies, use rt_yes unless
# modeling WATT3 and CCTHot as RT doesn't make sense in those paradigms
rt_mapping = {
    'stroop': ['rt_uncentered', 'no_rt'],
    'ANT': ['rt_uncentered', 'no_rt'],
    'CCTHot': ['no_rt'], 
    'stopSignal':['rt_uncentered', 'no_rt'], 
    'twoByTwo': ['rt_uncentered', 'no_rt'], 
    'WATT3': ['no_rt'],
    'discountFix': ['rt_uncentered', 'no_rt'], 
    'DPX': ['rt_uncentered', 'no_rt'], 
    'motorSelectiveStop': [ 'rt_uncentered', 'no_rt']
}

subids = get_subids(root)

for task in tasks:
    batch_root = Path(f'/oak/stanford/groups/russpold/data/uh2/aim1_mumford/'
                      f'output/{task}_lev1_output/batch_files/')
    batch_root.mkdir(parents=True, exist_ok=True)
    rt_options = rt_mapping[task]
    for rt_inc in rt_options:
        batch_file = (f'{batch_root}/task_{task}_rtmodel_{rt_inc}.batch')   
        with open(batch_stub) as infile, open(batch_file, 'w') as outfile:
            for line in infile:
                line = line.replace('JOBNAME', f"{task}_{rt_inc}")
                outfile.write(line)
            for sub in subids:
                outfile.write(
                    f"echo /oak/stanford/groups/russpold/data/uh2/"
                    f"aim1_mumford/rt_data_analysis/main_analysis_code/analyze_lev1.py {task} {sub} {rt_inc} \n"
                    f"/oak/stanford/groups/russpold/data/uh2/"
                    f"aim1_mumford/rt_data_analysis/main_analysis_code/analyze_lev1.py {task} {sub} {rt_inc} \n")





