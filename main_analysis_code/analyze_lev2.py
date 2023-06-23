#!/usr/bin/env python

import glob
import pandas as pd
import json
import numpy as np
import nibabel as nf
import shutil
from pathlib import Path
import stat 
import re
import sys   
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
  
rt_subset_dict = {
    'stroop': 'junk == False',
    'ANT': 'junk == False',
    'CCTHot': 'junk == False',
    'stopSignal': 'junk==False and trial_type != "stop_success"',
    'twoByTwo': 'junk==False',
    'WATT3': None,
    'discountFix': 'junk == False',
    'DPX': 'junk == False',
    'motorSelectiveStop': "junk == False & trial_type != 'crit_stop_success'"
}

rt_trial_grouping = {
    'stroop': ['trial_type'],
    'ANT': ['cue', 'flanker_type'],
    'CCTHot': None,
    'stopSignal': 'trial_type',
    'twoByTwo': ['CTI', 'cue_switch', 'task_switch'],
    'WATT3': None,
    'discountFix': ['trial_type'],
    'DPX': ['trial_type'],
    'motorSelectiveStop': ['trial_type']
}

rt_diff_definition = {
    'stroop': lambda rt_mean_by_group: float(rt_mean_by_group['incongruent'] - rt_mean_by_group['congruent']),
    'ANT': lambda rt_mean_by_group: float(((rt_mean_by_group['double_incongruent'] + rt_mean_by_group['spatial_incongruent'])/2  
            - (rt_mean_by_group['double_congruent'] + rt_mean_by_group['spatial_congruent'])/2)),
    'CCTHot': lambda rt_mean_by_group: None,
    'stopSignal': lambda rt_mean_by_group: float(rt_mean_by_group['stop_failure'] - rt_mean_by_group['go']),
    'twoByTwo': lambda rt_mean_by_group: float(rt_mean_by_group['900_task_switch'] - rt_mean_by_group['900_task_stay_cue_switch']),
    'WATT3': lambda rt_mean_by_group: None,
    'discountFix': lambda rt_mean_by_group: float(rt_mean_by_group['larger_later'] - rt_mean_by_group['smaller_sooner']),
    'DPX': lambda rt_mean_by_group: float(rt_mean_by_group['AY'] - rt_mean_by_group['BY']),
    'motorSelectiveStop': lambda rt_mean_by_group: float(rt_mean_by_group['crit_go'] - rt_mean_by_group['noncrit_signal'])
}


rt_diff_dv_checker = {
    'stroop': 'stroop_incong_minus_cong',
    'ANT': 'congruency_parametric',
    'CCTHot': None,
    'stopSignal': 'stop_failure-go',
    'twoByTwo': 'task_switch_cost_900',
    'WATT3': None,
    'discountFix': 'choice',
    'DPX': 'AY-BY',
    'motorSelectiveStop': 'crit_go-noncrit_nosignal'
}

contrast_definition_by_model = {
    'one_sampt': [['intercept']], 
    'rt_diff': [['intercept'], ['rt_diff']],
    'rt_diff_w_confounds': [['rt_diff']]
}


def get_subids(root):
    subdirs = glob.glob(f'{root}/s*/')
    subid = [val[-4:-1] for val in subdirs]
    return subid


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


def get_design_mat_row_subject(
    subid, task, root, rt_subset_dict, rt_trial_grouping, rt_diff_definition,
    model_lev2, confounds_btwn_sub
):
    confounds_this_sub = confounds_btwn_sub.loc[confounds_btwn_sub['index'] == f"s{subid}"]
    confounds_file = glob.glob(
        f'{root}/derivatives/fmriprep/sub-s{subid}/ses-[0-9]/func/*{task}*confounds*.tsv'
    )[0]
    confounds_within_sub= pd.read_csv(confounds_file, sep = '\t')
    events_tsv_file = glob.glob(
        f'{root}/sub-s{subid}/ses-[0-9]/func/*{task}*tsv'
    )[0]
    events_tsv = pd.read_csv(events_tsv_file, sep = '\t')
    if confounds_btwn_sub['index'].str.contains(f's{subid}').any():
        age = float(confounds_this_sub['age'])
        sex = int(confounds_this_sub['sex'])
        meanFD = confounds_within_sub['framewise_displacement'].mean()
    else:
        age = sex = meanFD = np.nan
    
    if 'rt' in model_lev2:
        events_for_rt = events_tsv.query(rt_subset_dict[task])
        rt_mean = events_for_rt['response_time'].mean()
        # This avoids setting with copy warning.  Jesus.
        for val in rt_trial_grouping[task]:
            new_col = events_for_rt[val].astype(str)
            events_for_rt = events_for_rt.drop(val, axis = 1)
            events_for_rt[val] = new_col

        rt_mean_by_group = events_for_rt.groupby(rt_trial_grouping[task])['response_time'].mean()
        rt_mean_by_group = rt_mean_by_group.to_frame().transpose()
        rt_mean_by_group.columns = \
            ['_'.join(col) if type(col) is tuple else col for col in rt_mean_by_group.columns.values]
        if task == 'twoByTwo':
            rename_dict = {
                '100.0_nan_switch': '100_task_switch',
                '100.0_stay_stay': '100_cue_stay',
                '100.0_switch_stay': '100_task_stay_cue_switch',
                '900.0_nan_switch': '900_task_switch',
                '900.0_stay_stay': '900_cue_stay',
                '900.0_switch_stay': '900_task_stay_cue_switch'
            }
            rt_mean_by_group.rename(columns=rename_dict, inplace=True)
        rt_diff = rt_diff_definition[task](rt_mean_by_group)
        if rt_diff == None:
            raise ValueError((f"Task {task} is not compatable "
                                f"with modelling RT"))
        designs = {
            'rt_diff': np.array([1, float(rt_diff)]),
            'rt_diff_w_confounds': np.array([1, float(rt_diff), age, sex, meanFD]),
            'all_rts_w_confounds': np.array([1] + 
                                list(rt_mean_by_group.loc['response_time']) + 
                                [age, sex, meanFD])
        }
        regressor_names = {
            'rt_diff': ['intercept', 'rt_diff'],
            'rt_diff_w_confounds': ['intercept', 'rt_diff', 'age', 'sex', 'meanFD'],
            'all_rts_w_confounds': ['intercept'] + list(rt_mean_by_group) + ['age', 'sex', 'meanFD'],
        }
    if 'rt' not in  model_lev2:
        designs = {'confounds_only': np.array([1, age, sex, meanFD])}
        regressor_names = {'confounds_only': ['intercept', 'age', 'sex', 'meanFD']}
    return designs[model_lev2], regressor_names[model_lev2]
    

def get_bold_and_sublist(lev1_task_contrast):
    task, lev1_contrast, rtmodel =  lev1_task_contrast.split(':')
    lev1_out = (f'/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/'
                f'{task}_lev1_output/task_{task}_rtmodel_{rtmodel}/')
    bold_files = sorted(
    glob.glob(f"{lev1_out}/contrast_estimates/*_contrast_{lev1_contrast}*rtmodel_{rtmodel}*")
    )
    sub_list = [re.search('_sub_(.*)_rtmodel_', val).group(1) for val in bold_files]
    return sub_list, bold_files


def build_desmat_all(
    lev1_task_contrast, model_lev2, root, rt_subset_dict, rt_trial_grouping, 
    rt_diff_definition, rt_diff_dv_checker
):
    task, lev1_contrast, rtmodel =  lev1_task_contrast.split(':')
    confounds_btwn_sub_file = (
        "/home/groups/russpold/uh2_analysis/Self_Regulation_Ontology_fMRI_2021/"
        "fmri_analysis/scripts/aim1_2ndlevel_regressors/"
        "aim1_2ndlevel_confounds_matrix.csv"
    )
    confounds_btwn_sub = pd.read_csv(confounds_btwn_sub_file)
    if model_lev2 == 'rt_diff' or model_lev2 == 'rt_diff_w_confounds':
        if lev1_contrast != rt_diff_dv_checker[task]:
            raise ValueError((f"Contrast {lev1_contrast} is not compatable "
                              f"with model {model_lev2}.  Only contrast "
                              f"{rt_diff_dv_checker[task]} can be used."))
    lev1_out = (f'/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/'
                f'{task}_lev1_output/task_{task}_rtmodel_{rtmodel}/')
    excluded_subjects = pd.read_csv(f'{lev1_out}/excluded_subject.csv')
    split_subid_task = pd.DataFrame(excluded_subjects.subid_task.str.split('_').to_list(), 
        columns=['subid', 'task'])
    excluded_subjects = pd.concat([split_subid_task, excluded_subjects], axis=1)
    excluded_subjects.drop('subid_task', inplace=True, axis=1)
    sub_list, bold_files = get_bold_and_sublist(lev1_task_contrast)
    desmat_all = []
    if model_lev2 != 'one_sampt':
        for subid in sub_list:    
            design, regressor_names = get_design_mat_row_subject(
                 subid, task, root, rt_subset_dict, rt_trial_grouping, 
                 rt_diff_definition, model_lev2, confounds_btwn_sub
            )
            desmat_all.append(design)
        desmat_all = np.array(desmat_all)
        rows_with_missing = np.isnan(desmat_all).any(axis = 1)
        desmat_final = desmat_all[~rows_with_missing, :]
        #desmat_final = desmat_final - desmat_final.mean(axis = 0, keepdims=True)
        #desmat_final[:,0] = 1
        bold_files_final = np.array(bold_files)[~rows_with_missing]
        if np.sum(rows_with_missing) > 0:
            summary_missing = pd.DataFrame(desmat_all[rows_with_missing, :],
                                        columns=regressor_names)
            summary_missing['subid'] = np.array(sub_list)[rows_with_missing]
            summary_missing['task'] = task
            summary_missing = summary_missing.reindex(columns=['subid']+['task'] +regressor_names)
            summary_missing.drop('intercept', axis=1, inplace=True)
            summary_missing = pd.merge(excluded_subjects, summary_missing, on =['subid', 'task'], how = 'outer')
        else:
            summary_missing = excluded_subjects    
    else:
        desmat_final = None
        regressor_names = None
        summary_missing = excluded_subjects
        bold_files_final = np.array(bold_files)
      
    return desmat_final, bold_files_final, regressor_names, summary_missing

  
def make_4d_data_mask(bold_files_final, outdir, lev1_task_contrast):
    from nilearn.maskers import NiftiMasker
    task, lev1_contrast, rtmodel =  lev1_task_contrast.split(':')
    data4d = nf.funcs.concat_images(bold_files_final)
    filename_root = (f'{outdir}/{task}_lev1_contrast_{lev1_contrast}_rtmod_{rtmodel}')
    data4d.to_filename(f'{filename_root}.nii.gz')

    mask = NiftiMasker().fit(data4d).mask_img_
    mask.to_filename(f'{filename_root}_mask.nii.gz')


def make_randomise_files(desmat_final, regressor_names, contrasts, outdir, model_lev2):
    from nilearn.glm.contrasts import expression_to_contrast_vector
    if model_lev2 != 'one_sampt':
        num_input_contrasts = desmat_final.shape[0]
        num_regressors = desmat_final.shape[1]
        desmat_path = f'{outdir}/desmat.mat'
        with open(desmat_path, 'w') as f:
            f.write(f'/NumWaves	{num_regressors} \n/NumPoints {num_input_contrasts} '
                    '\n/PPheights 1.000000e+00 \n \n/Matrix \n')
            np.savetxt(f, desmat_final, delimiter='\t')
        grp_path = f'{outdir}/desmat.grp' 
        with open(grp_path, 'w') as f:
            f.write(f'/NumWaves  1 \n/NumPoints {num_input_contrasts}\n \n/Matrix \n')
            np.savetxt(f, np.ones(num_input_contrasts), fmt='%s', delimiter='\t') 
    if model_lev2 == 'one_sampt':
        regressor_names = ['intercept']  
        num_regressors = 1     
    contrast_matrix = []
    num_contrasts = len(contrasts)
    for contrast in contrasts:
        contrast_def = expression_to_contrast_vector(
            contrast[0], regressor_names)
        contrast_matrix.append(np.array(contrast_def))
    con_path = f'{outdir}/desmat.con' 
    ppheight_and_reqeff = '\t '.join(str(val) for val in [1]*num_contrasts) 
    with open(con_path, 'w') as f:
        for val, contrast in enumerate(contrasts):
            f.write(f'/ContrastName{val+1} {contrast[0]}\n')
        f.write(f'/NumWaves  {num_regressors} \n/NumContrasts {num_contrasts}'
                f'\n/PPheights {ppheight_and_reqeff} ' 
                f'\n/RequiredEffect {ppheight_and_reqeff} \n \n/Matrix \n')
        np.savetxt(f, contrast_matrix, delimiter='\t') 
    fts_path = f'{outdir}/desmat.fts' 
    with open(fts_path, 'w') as f:
        f.write(f'/NumWaves  {num_contrasts} \n/NumContrasts {num_contrasts}\n \n/Matrix \n')
        np.savetxt(f, np.identity(num_contrasts), fmt='%s', delimiter='\t') 


def make_batch_file(outdir, model_lev2, lev1_task_contrast, batch_stub):
    task, lev1_contrast, rtmodel =  lev1_task_contrast.split(':')
    filename_input_root = (f'{outdir}/{task}_lev1_contrast_{lev1_contrast}_rtmod_{rtmodel}')
    if model_lev2=='one_sampt':
        randomise_call = (f'randomise -i {filename_input_root}.nii.gz'
                        f' -o {outdir}/randomise_output_model_{model_lev2} ' 
                        f'-m {filename_input_root}_mask.nii.gz'
                        f' -1 -t {outdir}/desmat.con' 
                        f' -f {outdir}/desmat.fts -T -n 5000')
    else:
        randomise_call = (f'randomise -i {filename_input_root}.nii.gz'
                        f' -o {outdir}/randomise_output_model_{model_lev2} '
                        f'-m {filename_input_root}_mask.nii.gz'
                        f' -d {outdir}/desmat.mat -t {outdir}/desmat.con' 
                        f' -f {outdir}/desmat.fts  -T -n 5000')
    randomise_call_file = Path(f'{outdir}/randomise_call.batch')
    with open(batch_stub) as infile, open(randomise_call_file, 'w') as outfile:
        for line in infile:
            line = line.replace('JOBNAME', f"{task}_lev2")
            line = line.replace('48', '5')
            outfile.write(line)
    with open (randomise_call_file, 'a') as f:
        f.write('module load contribs \n')
        f.write('module load poldrack \n')
        f.write('module load fsl \n')
        f.write(randomise_call)
    randomise_call_file.chmod(randomise_call_file.stat().st_mode | stat.S_IXGRP | stat.S_IEXEC)


def run_it(outdir, model_lev2, lev1_task_contrast):
    import subprocess
    task, lev1_contrast, rtmodel =  lev1_task_contrast.split(':')
    filename_input_root = (f'{outdir}/{task}_lev1_contrast_{lev1_contrast}_rtmod_{rtmodel}')
    if model_lev2=='one_sampt':
        randomise_call = (f'randomise -i {filename_input_root}.nii.gz'
                        f' -o {outdir}/randomise_output_model_{model_lev2} ' 
                        f'-m {filename_input_root}_mask.nii.gz'
                        f' -1 -t {outdir}/desmat.con' 
                        f' -f {outdir}/desmat.fts  -T -n 5000')
    else:  
        randomise_call = (f'randomise -i {filename_input_root}.nii.gz'
                        f' -o {outdir}/randomise_output_model_{model_lev2} ' 
                        f'-m {filename_input_root}_mask.nii.gz'
                        f' -d {outdir}/desmat.mat -t {outdir}/desmat.con' 
                        f' -f {outdir}/desmat.fts -e {outdir}/desmat.grp  -T -n 5000')
    script_file = Path(f'{outdir}/randomise_call.sh')
    with open (script_file, 'w') as f:
        f.write('#!/bin/bash \n')
        f.write('module load contribs \n')
        f.write('module load poldrack \n')
        f.write('module load fsl \n')
        f.write(randomise_call)
    script_file.chmod(script_file.stat().st_mode | stat.S_IXGRP | stat.S_IEXEC)
    script_out = subprocess.check_output(script_file)
    output_file = Path(f'{outdir}/randomise_output.txt')
    with open (output_file, 'w') as f:
        f.write(script_out.decode('ascii'))


def make_html_summary(
    desmat_final, bold_files_final, regressor_names, contrasts, 
    summary_missing, outdir, root
    ):
    import seaborn as sns
    from matplotlib import pyplot as plt
    from nilearn.glm.contrasts import expression_to_contrast_vector
    import base64
    from io import BytesIO
    from nilearn.plotting import plot_design_matrix, plot_carpet, plot_stat_map
    from nilearn import masking
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    num_missing = summary_missing.shape[0]
    num_con_est = len(bold_files_final)
    total_n = 110 - num_missing

    num_missing = summary_missing.shape[0]
    top_message = (f'<h1>Pre analysis summary</h1><br> <h2>Started with N=110, '
                       f'lost {num_missing} for various reasonse (see table), final '
                       f'Number of lower level contrasts = {num_con_est}.'
                       f'<br>If 110-(number expected missing) is not the same as '
                       f'the number of contrasts [{110-num_missing}?=?{num_con_est}],  '
                       f'check level 1 analyses.</h2>'
                       f'<br> <h2> Summary of who is missing and why </h2><br>')
    top_message_missing = summary_missing.round(decimals=3).transpose().to_html()

    desmat_pandas = pd.DataFrame(desmat_final, columns = regressor_names)
  
    if desmat_pandas.empty == False:
        cor_desmat = desmat_pandas.corr()
        correlation_matrix = cor_desmat.to_html()
        vif_data = pd.DataFrame()
        vif_data["feature"] = desmat_pandas.columns
        vif_data["VIF"] = [variance_inflation_factor(desmat_pandas.values, i)
                          for i in range(len(desmat_pandas.columns))]
        vif_table = vif_data.to_html()

        pairgrid = sns.PairGrid(data=desmat_pandas)
        pairgrid = pairgrid.map_upper(sns.scatterplot)
        pairgrid = pairgrid.map_diag(plt.hist)
        pairgrid = pairgrid.map_lower(sns.kdeplot, warn_singular=False)
        pairgrid_tmpfile = BytesIO()
        pairgrid.figure.savefig(pairgrid_tmpfile, format='png', dpi=80)
        pairgrid_encoded = base64.b64encode(pairgrid_tmpfile.getvalue()).decode('utf-8')
        html_pairgrid = f'<h2>Regressor summary plots</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(pairgrid_encoded) + '<br>'

        desmat_fig = plot_design_matrix(desmat_pandas)
        desmat_tmpfile = BytesIO()
        desmat_fig.figure.savefig(desmat_tmpfile, format='png', dpi=60)
        desmat_encoded = base64.b64encode(desmat_tmpfile.getvalue()).decode('utf-8')
        html_desmat = f'<h2>Design matrix</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(desmat_encoded) + '<br>'

        design_column_names = desmat_pandas.columns.tolist()
        contrast_matrix = []
        contrast_names = []
        for idx, contrast in enumerate(contrasts):
            contrast_names.append(f'contrast_{idx+1}')
            contrast_def = expression_to_contrast_vector(
                contrast[0], design_column_names)
            contrast_matrix.append(np.array(contrast_def))
        contrast_matrix = np.asmatrix(np.asarray(contrast_matrix))
        maxval = 1
        max_len = np.max([len(str(name)) for name in design_column_names])

        plt.figure(figsize=(.9 * len(design_column_names),
                            .5 * contrast_matrix.shape[0] + .2 * max_len))
        contrast_fig = plt.gca()
        mat = contrast_fig.matshow(contrast_matrix, aspect='equal',
                        cmap='gray', vmin=-maxval, vmax=maxval)
        contrast_fig.set_label('conditions')
        contrast_fig.set_ylabel('')
        contrast_fig.set_yticks(list(range(len(contrasts))), contrast_names)
        contrast_fig.xaxis.set(ticks=np.arange(len(design_column_names)))
        contrast_fig.set_xticklabels(design_column_names, rotation=50, ha='left')
        plt.colorbar(mat, fraction=0.025, pad=0.08, shrink=1)
        plt.tight_layout()
        contrast_tmpfile = BytesIO()
        contrast_fig.figure.savefig(contrast_tmpfile, format='png', dpi=75)
        contrast_encoded = base64.b64encode(contrast_tmpfile.getvalue()).decode('utf-8')
        html_contrast = f'<h2>Contrasts</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(contrast_encoded) + '<br>'

    if desmat_pandas.empty:
        vif_table = '<h2> no VIF</h2>'
        correlation_matrix = ''
        html_pairgrid = '<h2> no plots, design matrix is column of 1s'
        html_desmat = ' '
        html_contrast = ''

    data4d = nf.funcs.concat_images(bold_files_final)
    data4d_array = data4d.get_fdata()
    sub_list_final = [
        re.search('_sub_(.*)_rtmodel_', val).group(1) for val in bold_files_final
    ] 
    
    data_nonzero = data4d_array[data4d_array.nonzero()]
    cutoff = np.quantile(np.abs(data_nonzero), .99)
    mask_img = masking.compute_epi_mask(data4d)
    carpet = plot_carpet(data4d, mask_img, t_r=1, title='Raw data for all subjects',
                         cmap='Greys', vmin = -1*cutoff, vmax = cutoff, detrend=False)
    plt.xlabel('Subjects')
    carpet.figure.set_size_inches(20, 15)
    carpet_tmpfile = BytesIO()
    carpet.figure.savefig(carpet_tmpfile, format='png', dpi=80)
    carpet_encoded = base64.b64encode(carpet_tmpfile.getvalue()).decode('utf-8')
    html_carpet = f'<h2>Brain data summary</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(carpet_encoded) + '<br>'

    ncols = 10
    nrows = 10

    brain_grid, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 22))
    brain_grid.suptitle("Slice 40 of brain for each subject", fontsize=15)
    for idx, ax in enumerate(axs.ravel()):
        if idx < total_n:
            ax.set_title(f"subject {sub_list_final[idx]}", fontsize=10)
            ax.imshow(np.flipud(np.transpose(data4d_array[:, :, 40, idx])), cmap='Greys',
                vmin = -1*cutoff, vmax = cutoff, aspect='auto')
            ax.axis('off')
        if idx >= total_n:
            ax.set_title('blank', fontsize=10)
            ax.imshow(np.zeros(data4d_array[:, :, 40, 1].shape), cmap='Greys',
                vmin = -1*cutoff, vmax = cutoff)
            ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    brain_grid_tmpfile = BytesIO()
    brain_grid.figure.savefig(brain_grid_tmpfile, format='png', dpi=80)
    brain_grid_encoded = base64.b64encode(brain_grid_tmpfile.getvalue()).decode('utf-8')
    html_brain_grid = (f'<h2>Single slice of data for each subject</h2>' + 
       '<img src=\'data:image/png;base64,{}\'>'.format(brain_grid_encoded) + '<br>')

    with open(f'{outdir}/model_summary.html','w') as f:
        f.write(top_message)
        f.write(top_message_missing)
        f.write(html_pairgrid)
        f.write('<h2>Correlation between regressors</h2><br>')
        f.write(correlation_matrix)
        f.write('<h2>Variance inflation factors</h2><br>')
        f.write(vif_table)
        f.write(html_desmat)
        f.write(html_contrast) 
        f.write(html_carpet) 
        f.write(html_brain_grid)
  

def get_parser():
    """Build parser object"""
    parser = ArgumentParser(
        prog='setup_randomise_lev2',
        description=('setup_randomise_lev2: Sets up randomise analysis with '
            'option to run.  Contrast for overall mean is always included. '
            'The rt_diff models will also include a contrast for the rt_diff. '
            'Two-sided t-tests (aka 1DF F-tests) will be run using 5000 permutations.'
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        'lev1_task_contrast',
        choices=check_valid_contrasts(),
        action='store',
        help="Use to specify task and contrast.",
    )
    parser.add_argument(
        'model_lev2',
        choices=['one_sampt', 'rt_diff', 'rt_diff_w_confounds'],
        action='store',
        help=("Use to specify model. rt_diff and rt_diff_w_confounds only works "
              "with congruency_parametric, stop_failure-go, task_switch_cost_900, "
              "AY-BY, crit_go-noncrit_signal.  Intercept is always included."
        ),
    )
    return parser
  
 
if __name__ == "__main__":
    argv = sys.argv[1:]
    opts = get_parser().parse_args(argv)
    lev1_task_contrast = opts.lev1_task_contrast
    model_lev2 = opts.model_lev2

    batch_stub = '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/rt_data_analysis/main_analysis_code/run_stub.batch'
    root = '/oak/stanford/groups/russpold/data/uh2/aim1/BIDS'
    task, lev1_contrast, rtmodel =  lev1_task_contrast.split(':')

    outdir = Path(f"/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/"
              f"{task}_lev2_output/{task}_lev1_contrast_{lev1_contrast}_rtmod_{rtmodel}_"
              f"lev2_model_{model_lev2}_new_omission/")
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    desmat_final, bold_files_final, regressor_names, summary_missing = build_desmat_all(
    lev1_task_contrast, model_lev2, root, rt_subset_dict, rt_trial_grouping, 
    rt_diff_definition, rt_diff_dv_checker
    )
    contrasts = contrast_definition_by_model[model_lev2]
    make_html_summary(
        desmat_final, bold_files_final, regressor_names, contrasts, 
        summary_missing, outdir, root
    )
    make_4d_data_mask(bold_files_final, outdir, lev1_task_contrast)
    make_randomise_files(desmat_final, regressor_names, contrasts, outdir, model_lev2)
    make_batch_file(outdir, model_lev2, lev1_task_contrast, batch_stub)
    #run_it(outdir, model_lev2, lev1_task_contrast)



    
 

