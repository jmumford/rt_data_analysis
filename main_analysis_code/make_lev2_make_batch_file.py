#!/usr/bin/env python

import sys
sys.path.insert(1, '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/rt_data_analysis/main_analysis_code')
from analyze_lev2 import check_valid_contrasts

all_cons = check_valid_contrasts()

batch_stub = ('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/'
              'rt_data_analysis/main_analysis_code/run_stub.batch')
batch_outfile = ('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/'
              'rt_data_analysis/main_analysis_code/make_lev2_batch_files.batch')

all_cons = ['DPX:AY_cue+AY_probe-BY_cue-BY_probe:rt_duration', 
            'DPX:response_time:rt_duration']
with open(batch_stub) as infile, open(batch_outfile, 'w') as outfile:
    for line in infile:
        line = line.replace('JOBNAME', 'batch_maker')
        outfile.write(line)
    for con in all_cons:
        if 'DPX' in con:
            outfile.write(
                f"echo /oak/stanford/groups/russpold/data/uh2/"
                f"aim1_mumford/rt_data_analysis/main_analysis_code/analyze_lev2.py {con} one_sampt \n"
                f"/oak/stanford/groups/russpold/data/uh2/"
                f"aim1_mumford/rt_data_analysis/main_analysis_code/analyze_lev2.py {con} one_sampt  \n")


