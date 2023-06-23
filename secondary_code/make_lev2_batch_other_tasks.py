import re
import shutil


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


task_con_mod_combos = check_valid_contrasts()

batch_stub = ('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/'
              'rt_data_analysis/secondary_code/group_stub.batch')

batch_other_tasks = ('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/'
              'rt_data_analysis/secondary_code/group_non_stroop.batch')

for task_con_mod_combo in task_con_mod_combos:
    if ('stroop' not in task_con_mod_combo and 'WATT3' not in task_con_mod_combo 
        and 'CCTHot' not in task_con_mod_combo 
        and ':task:' not in task_con_mod_combo and 'response_time' not in task_con_mod_combo):
        print(task_con_mod_combo)

shutil.copy(batch_stub, batch_other_tasks)
with open(batch_stub) as infile, open(batch_other_tasks, 'a') as outfile:
    for task_con_mod_combo in task_con_mod_combos:
        if ('stroop' not in task_con_mod_combo and 'WATT3' not in task_con_mod_combo 
            and 'CCTHot' not in task_con_mod_combo 
            and 'task' not in task_con_mod_combo and 'response_time' not in task_con_mod_combo):
            outfile.write(f'echo /oak/stanford/groups/russpold/data/uh2/aim1_mumford/rt_data_analysis/main_analysis_code/analyze_lev2.py '
                f'{task_con_mod_combo} one_sampt \n')
            outfile.write(f'/oak/stanford/groups/russpold/data/uh2/aim1_mumford/rt_data_analysis/main_analysis_code/analyze_lev2.py '
                f'{task_con_mod_combo} one_sampt \n')
