from nilearn.plotting import plot_design_matrix
from nilearn.glm.contrasts import expression_to_contrast_vector
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from pathlib import Path


def get_behav_exclusion(subid, task):
    """
    Loads in behavioral exclusion file and extracts data for given task and subject
    input:
        subid: Subject ID
        task: task name
    output: pandas data frame with any behavioral exclusions for this subject/task
    """
    behav_exclusion_file = ('/oak/stanford/groups/russpold/data/uh2/'
        'aim1_mumford/rt_data_analysis/main_analysis_code/utils_lev1/aim1_beh_subject_exclusions.csv')
    behav_exclusion = pd.read_csv(behav_exclusion_file)
    behav_exclusion.rename(columns={'Unnamed: 0': 'subid_task'}, inplace=True)
    behav_exclusion['subid_task'] = behav_exclusion['subid_task'].map(lambda x: x.lstrip('s'))
    behav_exclusion['subid_task'] = behav_exclusion['subid_task'].str.replace('WATT', 'WATT3')
    behav_exclusion_this_sub = behav_exclusion[behav_exclusion['subid_task'].str.contains(f'{subid}_{task}')]
    if behav_exclusion_this_sub.empty:
        behav_exclusion_this_sub = pd.DataFrame((np.atleast_2d([f'{subid}_{task}']+ [0]*7)), 
            columns = list(behav_exclusion_this_sub.columns))   
        cols_to_int = [val for val in behav_exclusion_this_sub.columns if val not in ['subid_task']]
        for col in cols_to_int:
            behav_exclusion_this_sub[col] = pd.to_numeric(behav_exclusion_this_sub[col])     
    return behav_exclusion_this_sub


def qa_design_matrix(contrast_dir, contrasts, desmat, subid, task, percent_junk, percent_high_motion):
    """
    Check design matrix for regressors that are included in contrasts that have 
    all zeros. >20% censored volumes (motion), >10% junk trials and unusually low
    number of TRs 
    input:
      contrast_dir: output contrast directory
      contrasts: contrasts to be estimated
      desmat:  design matrix
      subid: subject id number (without 's')
      task: task name 
      percent_junk: percent of junk trials (calculated when design matrix is made)
      pecent_high_motion: percent of high motion volumes (calculated when confounds are extracted)
    return:
      any_fail: True=design is good to go, False=skip this run due to QA failures
      error_message: Message explaining why subject was excluded (written to file as well)
      If errors are found, the excluded.csv file is updated
    """
    import functools as ft
    num_time_point_cutoff = {
        'motorSelectiveStop': 800,
        'DPX': 1000,
        'stroop': 300,
        'discountFix': 800,
        'CCTHot': 500,
        'ANT': 500,
        'WATT3': 300,
        'stopSignal': 400,
        'twoByTwo': 800
    }
    behav_exclusion_this_sub = get_behav_exclusion(subid, task)
    design_column_names = desmat.columns.tolist()
    contrast_matrix = []
    for i, (key, values) in enumerate(contrasts.items()):
        contrast_def = expression_to_contrast_vector(
            values, design_column_names)
        contrast_matrix.append(np.array(contrast_def))
    columns_to_check = np.where(np.sum(np.abs(contrast_matrix), 0)!=0)[0]
    checked_columns_fail = (desmat.iloc[:,columns_to_check] == 0).all()
    any_column_fail = checked_columns_fail.any()
    bad_columns = list(checked_columns_fail.index[checked_columns_fail.values])
    bad_columns = '_and_'.join(str(x) for x in bad_columns)
 
    #num_trs = desmat.shape[0]

    failures = {'subid_task': f'{subid}_{task}',
                'percent_junk_gt_45': [percent_junk if percent_junk > .45 else 0],
                'percent_scrub_gt_20': [percent_high_motion if percent_high_motion > .2 else 0],
                #f'num_trs_lt_{num_time_point_cutoff[task]}': [num_trs if num_trs < num_time_point_cutoff[task] else 0],
                'task_related_regressor_all_zeros': bad_columns if any_column_fail else [0]}
    failures = pd.DataFrame(failures)
    all_exclusion = pd.merge(behav_exclusion_this_sub, failures)
    any_fail = all_exclusion.loc[:, all_exclusion.columns != 'subid_task'].ne(0).any(1).bool()
    if any_fail:
        update_excluded_subject_csv(all_exclusion, subid, task, contrast_dir)
    return all_exclusion, any_fail


def est_vif(desmat):
    """"
    General variance inflation factor estimation.  Calculates VIF for all 
    regressors in the design matrix
    input:
        desmat: design matrix.  Intercept not required.
    output:
      vif_data: Variance inflation factor for each regressor in the design matrix
                generally goal is VIF<5
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    desmat_with_intercept = desmat.copy()
    desmat_with_intercept['intercept'] = 1
    vif_data = pd.DataFrame()
    vif_data['regressor'] = desmat_with_intercept.columns.drop('intercept')
    vif_data['VIF'] = [variance_inflation_factor(desmat_with_intercept.values, i)
                          for i in range(len(desmat_with_intercept.columns))
                          if desmat_with_intercept.columns[i] != 'intercept']
    return vif_data


def get_eff_reg_vif(desmat, contrast):
    """"
    The goal of this function is to estimate a variance inflation factor for a contrast.
    This is done by extending the effective regressor definition from Smith et al (2007)
    Meaningful design and contrast estimability (NeuroImage).  Regressors involved
    in the contrast estimate are rotated to span the same space as the original space
    consisting of the effective regressor and and an orthogonal basis.  The rest of the 
    regressors are unchanged.
    input:
        desmat: design matrix.  Assumed to be a pandas dataframe with column  
             headings which are used define the contrast of interest
        contrast: a single contrast defined in string format
    output:
        vif: a single VIF for the contrast of interest  
    """
    from scipy.linalg import null_space
    from nilearn.glm.contrasts import expression_to_contrast_vector
    contrast_def = expression_to_contrast_vector(contrast, desmat.columns)
    des_nuisance_regs = desmat[desmat.columns[contrast_def == 0]]
    des_contrast_regs = desmat[desmat.columns[contrast_def != 0]]

    con = np.atleast_2d(contrast_def[contrast_def != 0])
    con2_t = null_space(con)
    con_t = np.transpose(con)
    x = des_contrast_regs.copy().values
    q = np.linalg.pinv(np.transpose(x)@ x)
    f1 = np.linalg.pinv(con @ q @ con_t)
    pc = con_t @ f1 @ con @ q
    con3_t = con2_t - pc @ con2_t
    f3 = np.linalg.pinv(np.transpose(con3_t) @ q @ con3_t)
    eff_reg = x @ q @ np.transpose(con) @ f1
    eff_reg = pd.DataFrame(eff_reg, columns = [contrast])

    other_reg = x @ q @ con3_t @ f3 
    other_reg_names = [f'orth_proj{val}' for val in range(other_reg.shape[1])]
    other_reg = pd.DataFrame(other_reg, columns = other_reg_names)

    des_for_vif = pd.concat([eff_reg, other_reg, des_nuisance_regs], axis = 1)
    vif_dat = est_vif(des_for_vif)
    vif_dat.rename(columns={'regressor': 'contrast'}, inplace=True)
    vif_output = vif_dat[vif_dat.contrast == contrast]
    return vif_output


def get_all_contrast_vif(desmat, contrasts):
    """
    Calculates the VIF for multiple contrasts
    input:
        desmat: design matrix.  Pandas data frame, column names must 
                be used in the contrast definitions
        contrasts: A dictionary of contrasts defined in string format
    output:
        vif_contrasts: Data frame containing the VIFs for all contrasts
    """
    vif_contrasts = {'contrast': [],
                      'VIF': []}
    for key, item in contrasts.items():
        vif_out = get_eff_reg_vif(desmat, item)
        vif_contrasts['contrast'].append(vif_out['contrast'][0])
        vif_contrasts['VIF'].append(vif_out['VIF'][0]) 
    vif_contrasts = pd.DataFrame(vif_contrasts)
    return vif_contrasts  


def check_html_for_sub(subid, html_file):
    """
    Checks whether the sub info has already been added to html
    input:
        subid: subject ID
        html_file: path to html output file
    output:
        already_done: T/F indicating if this subject was already done
    """
    html_file_pth = Path(html_file)
    already_done = False
    if html_file_pth.exists():
        with open(html_file_pth) as myfile:
            if f'subject {subid}' in myfile.read():
                already_done = True
    return already_done



def add_to_html_summary(subid, contrasts, desmat, outdir, regress_rt, task, any_fail, exclusion):
    """
    Adds QA summaries to html file.  A single file per task is generated
    input:
        subid: Subject ID
        contrasts: Contrasts of interest
        desmat: Design matrix
        outdir: Output directory
        regress_rt:  How RT was modeled
        task: Task
        any_fail: Whether any QA checks failed
        exclusion: Pandas dataframe indicating which exclusionary criteria were met
    ouput:
        HTML file is updated
    """
    html_file = (f'{outdir}/contrasts_task_{task}_rtmodel_{regress_rt}_model_summary.html') 
    sub_already_done = (subid, html_file)
    if sub_already_done == False:
        desmat_fig = plot_design_matrix(desmat)
        desmat_tmpfile = BytesIO()
        desmat_fig.figure.savefig(desmat_tmpfile, format='png', dpi=60)
        desmat_encoded = base64.b64encode(desmat_tmpfile.getvalue()).decode('utf-8')
        if not any_fail:
            html_desmat = f'<h2>{task} design for subject {subid}</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(desmat_encoded) + '<br>'
        if any_fail:
            html_desmat = f'<h2>FAIL, analysis skipped! <br> {exclusion.T.to_html()} <br> {task} design for subject {subid}</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(desmat_encoded) + '<br>'

        design_column_names = desmat.columns.tolist()
        contrast_matrix = []
        for i, (key, values) in enumerate(contrasts.items()):
            contrast_def = expression_to_contrast_vector(
                values, design_column_names)
            contrast_matrix.append(np.array(contrast_def))
        contrast_matrix = np.asmatrix(np.asarray(contrast_matrix))
        #maxval = np.max(np.abs(contrast_def))
        maxval = 1
        max_len = np.max([len(str(name)) for name in design_column_names])

        plt.figure(figsize=(.4 * len(design_column_names),
                                1 + .5 * contrast_matrix.shape[0] + .1 * max_len))
        contrast_fig = plt.gca()
        mat = contrast_fig.matshow(contrast_matrix, aspect='equal',
                        cmap='gray', vmin=-maxval, vmax=maxval)
        contrast_fig.set_label('conditions')
        contrast_fig.set_ylabel('')
        contrast_fig.set_yticks(list(range(len(contrasts))), list(contrasts.keys()))
        contrast_fig.xaxis.set(ticks=np.arange(len(design_column_names)))
        contrast_fig.set_xticklabels(design_column_names, rotation=50, ha='left')
        plt.colorbar(mat, fraction=0.025, pad=0.08, shrink=.5)
        plt.tight_layout()
        contrast_tmpfile = BytesIO()
        contrast_fig.figure.savefig(contrast_tmpfile, format='png', dpi=75)
        contrast_encoded = base64.b64encode(contrast_tmpfile.getvalue()).decode('utf-8')
        html_contrast = f'<h2>{task} contrasts for subject {subid}</h2>' + '<img src=\'data:image/png;base64,{}\'>'.format(contrast_encoded) + '<br>'
    
        desmat_vif = desmat
        vif_data = est_vif(desmat_vif)
        vif_data_table = vif_data[~vif_data.regressor.str.contains(r'(?:reject|trans|rot|comp_cor|non_steady)')]
        vif_table = vif_data_table.to_html(index = False)

        vif_contrasts = get_all_contrast_vif(desmat, contrasts)
        vif_contrasts_table = vif_contrasts.to_html(index = False)

        corr_matrix = desmat.corr()
        f,  heatmap= plt.subplots(figsize=(20,20)) 
        heatmap = sns.heatmap(corr_matrix, 
                        square = True,
                        vmin=-1, vmax=1, center=0,
                        cmap="coolwarm")
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )
        cormat_tmpfile = BytesIO()
        heatmap.figure.savefig(cormat_tmpfile, format='png', dpi=60)
        cormat_encoded = base64.b64encode(cormat_tmpfile.getvalue()).decode('utf-8')
        html_cormat = '<img src=\'data:image/png;base64,{}\'>'.format(cormat_encoded) + '<br>'
        
        with open(html_file,'a') as f:
            f.write('<hr>')
            f.write(f'<h2>Subject {subid}</h2><br>')
            f.write(html_desmat)
            f.write(html_contrast) 
            f.write(f'<h2>Variance inflation factors subject {subid}</h2><br>')
            f.write(vif_table)
            f.write(vif_contrasts_table)
            f.write(html_cormat)
        plt.close('all')


def update_excluded_subject_csv(current_exclusion, subid, task, contrast_dir):
    """
    Reads in previously calculated behavioral exclusions and combines with the
    current_exclusion data, which were exclusionary criteria determined after behavioral
    QA.  Only called within design_matrix_qa
    input:
        current_exclusion: Exclusion criteria from within design_matrix_qa
        subid: subject ID
        task: task
        contrast_dir: Directory where contrasts are to be saved
    ouput: 
      Appends exclusion information to a master csv file in the contrast directory. 
    """
    import functools as ft
    behav_exclusion_this_sub = get_behav_exclusion(subid, task)
    full_sub_exclusion = pd.merge(behav_exclusion_this_sub, current_exclusion, how='inner')
    exclusion_out_path = Path(f'{contrast_dir}/excluded_subject.csv')
    if exclusion_out_path.exists():
        old_exclusion_csv = pd.read_csv(exclusion_out_path)
        full_sub_exclusion['subid_task'] = f'{subid}_{task}'
        old_and_new_exclusion = pd.concat([old_exclusion_csv, full_sub_exclusion], axis = 0)
        old_and_new_exclusion.fillna(0, inplace=True)
        #just in case this is run more than once
        old_and_new_exclusion.drop_duplicates(inplace=True)
    else:
        old_and_new_exclusion = current_exclusion
    old_and_new_exclusion.to_csv(exclusion_out_path, index=False)
