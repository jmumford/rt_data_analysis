import glob
import pandas as pd
import json
import numpy as np
import nibabel as nf
import shutil
from pathlib import Path
import re


def check_valid_contrasts():
    """Used to keep potential contrast names up to date in parser
       Does require subject 497 has been analyzed.  Probably a more stable way
       to do this.
    """
    import glob
    possible_task_contrasts = glob.glob(
        "/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/"
        "*_lev1_output/*/contrast*/*contrast*"
    )
    task_con_names = []
    
    for contrast in possible_task_contrasts:
        contrast_file = contrast.split('/')[-1]
        con_name = re.search('_contrast_(.*)_sub_', contrast_file).group(1)
        task_name = re.search('task_(.*)_contrast_', contrast_file).group(1)
        rt_model = re.search('_rtmodel_(.*)_stat', contrast_file).group(1)
        task_con_names.append(f"{task_name}:{con_name}:{rt_model}")
    task_con_names = np.ndarray.tolist(np.unique(task_con_names))
    return task_con_names        



def get_bold_and_sublist(lev1_task_contrast):
    task, lev1_contrast, rtmodel =  lev1_task_contrast.split(':')
    lev1_out = (f'/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/'
                f'{task}_lev1_output/task_{task}_rtmodel_{rtmodel}/')
    bold_files = sorted(
    glob.glob(f"{lev1_out}/contrast_estimates/*_contrast_{lev1_contrast}*rtmodel_{rtmodel}*")
    )
    sub_list = [re.search('_sub_(.*)_rtmodel_', val).group(1) for val in bold_files]
    return sub_list, bold_files




check = ['ANT:congruency_parametric:rt_duration', 
         'DPX:AY_cue+AY_probe-BY_cue-BY_probe:rt_duration',
         'discountFix:task:rt_duration',
         'motorSelectiveStop:response_time:rt_duration',
         'stopSignal:stop_success-go:rt_duration',
         'stroop:response_time:rt_duration',
         'twoByTwo:task_switch_cost_100:rt_duration']

sub_info_file = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS/participants.tsv'
sub_info = pd.read_csv(sub_info_file, sep='\t')



for task in check:
    sub_list, _ = get_bold_and_sublist(task)
    sub_list_formatted = [f'sub-s{val}' for val in sub_list]
    sub_info_task = sub_info[sub_info['participant_id'].isin(sub_list_formatted)]
    mn_age = np.round(np.mean(sub_info_task['Age (years)']), 1)
    sd_age = np.round(np.std(sub_info_task['Age (years)']), 1)
    sum_female = np.sum(sub_info_task['Gender']=='Female')
    print(task)
    print(f'mean age, {mn_age}')
    print(f'std age, {sd_age}')
    print(f'female N, {sum_female}')






    for sub in sub_list_formatted:
        if not sub_info['participant_id'].isin([sub]).any():
            print(sub)
            