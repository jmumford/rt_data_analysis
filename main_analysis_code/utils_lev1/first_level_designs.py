from nilearn.glm.first_level import compute_regressor
import numpy as np
import pandas as pd



def make_regressor_and_derivative(n_scans, tr, events_df, add_deriv,
                   amplitude_column=None, duration_column=None,
                   onset_column=None, subset=None, demean_amp=False, 
                   cond_id = 'cond'):
    """ Creates regressor and derivative using spm + derivative option in
        nilearn's compute_regressor
        Input:
          n_scans: number of scans
          tr: time resolution in seconds
          events_df: events data frame
          add_deriv: "yes"/"no", whether or not derivatives of regressors should
                     be included
          amplitude_column: Required.  Amplitude column from events_df
          duration_column: Required.  Duration column from events_df
          onset_column: optional.  if not specified "onset" is the default
          subset: optional.  Boolean for subsetting rows of events_df
          demean_amp: Whether amplitude should be mean centered
          cond_id: Name for regressor that is created.  Note "cond_derivative" will
            be assigned as name to the corresponding derivative
        Output:
          regressors: 2 column pandas data frame containing main regressor and derivative
    """
    if subset == None:
        events_df['temp_subset'] = True
        subset = 'temp_subset == True'
    if onset_column == None:
        onset_column = 'onset'
    if amplitude_column == None or duration_column == None:
        print('Must enter amplitude and duration columns')
        return
    if amplitude_column not in events_df.columns:
        print("must specify amplitude column that exists in events_df")
        return
    if duration_column not in events_df.columns:
        print("must specify duration column that exists in events_df")
        return
    
    reg_3col = events_df.query(subset)[[onset_column, duration_column, amplitude_column]]
    reg_3col = reg_3col.rename(
        columns={duration_column: "duration",
        amplitude_column: "modulation"})
    if demean_amp:
        reg_3col['modulation'] = reg_3col['modulation'] - \
        reg_3col['modulation'].mean()
    if add_deriv == 'deriv_yes':
        hrf_model = 'spm + derivative'
    else:
        hrf_model= 'spm'
        
    regressor_array, regressor_names = compute_regressor(
        np.transpose(np.array(reg_3col)),
        hrf_model,
        np.arange(n_scans)*tr+tr/2,
        con_id=cond_id
    ) 
    regressors =  pd.DataFrame(regressor_array, columns=regressor_names)  
    return regressors


def define_nuisance_trials(events_df, task):
    """
    Splits junk trials into omission, commission and too_fast, with the exception
    of twoByTwo where too_fast alsoo includes first trial of block
    Note, these categories do not apply to WATT3 or CCTHot
    inputs: 
        events_df: the pandas events data frame
        task: The task name
    output:
        too_fast, omission, commission: indicators for each junk trial type
    """
    if task in ['ANT', 'DPX', 'stroop']:
        omission = (events_df.key_press == -1)
        commission = ((events_df.key_press != events_df.correct_response) &
                      (events_df.key_press != -1) &
                      (events_df.response_time >= .2))
        too_fast = (events_df.response_time < .2) 
    if task in ['twoByTwo']:
        omission = (events_df.key_press == -1)
        commission = ((events_df.key_press != events_df.correct_response) &
                      (events_df.key_press != -1) & 
                      (events_df.response_time >= .2))
        too_fast = ((events_df.response_time < .2) |
                    (events_df.first_trial_of_block == 1))
    if task in ['stopSignal']:
        omission = ((events_df.trial_type == 'go') &
                    (events_df.key_press == -1))
        commission = ((events_df.trial_type == 'go') &
                      (events_df.key_press != events_df.correct_response) &
                      (events_df.response_time >= .2))
        too_fast = ((events_df.trial_type == 'go') &
                    (events_df.key_press != -1) &
                    (events_df.response_time < .2))
    if task in ['motorSelectiveStop']:
        trial_type_list = ['crit_go', 'noncrit_nosignal', 'noncrit_signal']
        omission = ((events_df.trial_type.isin(trial_type_list)) &
                    (events_df.key_press == -1))
        commission = ((events_df.trial_type.isin(trial_type_list)) &
                      (events_df.key_press != events_df.correct_response) &
                      (events_df.response_time >= .2))
        too_fast = ((events_df.trial_type.isin(trial_type_list)) &
                    (events_df.key_press != -1) &
                    (events_df.response_time < .2))
    if task in ['discountFix']:  
        omission = (events_df.key_press == -1)
        commission = 0*omission
        too_fast = (events_df.response_time < .2)
    events_df['omission'] = 1 * omission
    events_df['commission'] = 1 * commission
    events_df['too_fast'] = 1 * too_fast
    percent_junk = np.mean(omission | commission | too_fast)
    return events_df, percent_junk


def make_basic_stroop_desmat(
    events_file, add_deriv, regress_rt, n_scans, tr, 
    confound_regressors
):
    """Creates basic stroop regressors (and derivatives) 
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk =  define_nuisance_trials(events_df, 'stroop')
    events_df['constant_1_column'] = 1 
    events_df['incongruent'] = 0
    events_df.loc[events_df.trial_type == 'incongruent', 'incongruent'] = 1
    events_df['congruent'] = 0
    events_df.loc[events_df.trial_type == 'congruent', 'congruent'] = 1
    subset_main_regressors = 'too_fast == 0 and commission == 0 and omission == 0 and onset > 0' 
    subset_main_regressors_congruent = 'too_fast == 0 and commission == 0 and omission == 0 and onset > 0 and congruent == 1'
    subset_main_regressors_incongruent = 'too_fast == 0 and commission == 0 and omission == 0 and onset > 0 and incongruent == 1'

    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'too_fast'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'commission'
        )
    congruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="congruent", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id ='congruent'
    )
    incongruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="incongruent", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='incongruent'
    )
    design_matrix = pd.concat([congruent, incongruent,
        too_fast_regressor, omission_regressor, commission_regressor, confound_regressors], axis=1)
    contrasts = {
        "stroop_incong_minus_cong": "incongruent - congruent",
        #"stroop_cong_v_baseline": "congruent",
        #"stroop_incong_v_baseline":"incongruent"
        }
    if regress_rt == 'rt_centered':
        mn_rt = events_df.query(subset_main_regressors)['response_time'].mean()
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration_only':
        rt_congruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors_congruent, demean_amp=False, cond_id='congruent_rtdur'
        ) 
        rt_incongruent = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors_incongruent, demean_amp=False, cond_id='incongruent_rtdur'
        )
        design_matrix = pd.concat([rt_congruent, rt_incongruent,
        too_fast_regressor, omission_regressor, commission_regressor, confound_regressors], axis=1)
        contrasts = {
        "stroop_incong_rtdur_minus_cong_rtdur": "incongruent_rtdur - congruent_rtdur"
        }
    
    return design_matrix, contrasts, percent_junk


def make_basic_ant_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic ANT regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Returns
          design_matrix: pd data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'ANT')
    subset_main_regressors = ('too_fast == 0 and commission == 0'
                              'and omission == 0 and onset > 0') 
    events_df['constant_1_column'] = 1

    events_df['cue_parametric'] = 0
    events_df.loc[events_df.cue == 'double', 'cue_parametric'] = 1
    events_df.loc[events_df.cue == 'spatial', 'cue_parametric'] = -1

    events_df['congruency_parametric'] = 0
    events_df.loc[events_df.flanker_type == 'incongruent', 'congruency_parametric'] = 1
    events_df.loc[events_df.flanker_type == 'congruent', 'congruency_parametric'] = -1

    events_df['cue_congruency_interaction'] = events_df.cue_parametric.values *\
                                              events_df.congruency_parametric.values
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'too_fast'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'commission'
        )
    cue_parametric = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="cue_parametric", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp = True, cond_id = 'cue_parametric'
    )
    congruency_parametric = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="congruency_parametric", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=True, cond_id='congruency_parametric'
    )
    cue_congruency_interaction = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="cue_congruency_interaction", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=True, cond_id='interaction'
    )
    all_trials = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='task'
    )
    design_matrix = pd.concat([cue_parametric, congruency_parametric,
        cue_congruency_interaction, all_trials, too_fast_regressor, omission_regressor,
        commission_regressor, confound_regressors], axis=1)
    contrasts = {'cue_parametric': 'cue_parametric',
                'congruency_parametric': 'congruency_parametric',
                #'interaction': 'interaction',#
                #'task': 'task'#
                }
    if regress_rt == 'rt_centered':
        mn_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk


def make_basic_ccthot_desmat(events_file, add_deriv, regress_rt, 
    n_scans, tr, confound_regressors
):
    """Creates basic CCTHot regressors (and derivatives)
       Input:
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return:
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    # no junk trial definition for this task
    percent_junk = 0
    events_df['constant_1_column'] = 1  
    end_round_idx = events_df.index[events_df.trial_id == 'ITI']
    # shift by 1 to next trial start, ignoring the last ITI
    start_round_idx = [0] + [x+1 for x in end_round_idx[:-1]]
    assert len(end_round_idx) == len(start_round_idx)
    events_df['trial_start'] = False
    events_df.loc[start_round_idx, 'trial_start'] = True

    trial_durs = []
    for start_idx, end_idx in zip(start_round_idx, end_round_idx):
            # Note, this automatically excludes the ITI row
        trial_durs.append(
            events_df.iloc[start_idx:end_idx]
                            ['block_duration'].sum()
        )
    events_df['trial_duration'] = np.nan
    events_df.loc[start_round_idx, 'trial_duration'] = trial_durs
    
    all_task = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="trial_duration",
        subset='trial_start==True and onset > 0', demean_amp=False, cond_id='task'
    )
    events_df['button_onset'] = events_df.onset+events_df.response_time
    pos_draw = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="EV", duration_column="constant_1_column",
        onset_column='button_onset',
        subset='trial_start==True', demean_amp=True, 
        cond_id='positive_draw'
    )
    events_df['absolute_loss_amount'] = np.abs(events_df.loss_amount)
    neg_draw = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="absolute_loss_amount", duration_column="constant_1_column",
        onset_column='button_onset',
        subset="action=='draw_card' and feedback==0 and onset > 0", demean_amp=True, 
        cond_id='negative_draw'
    )
    trial_gain = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="gain_amount", duration_column="trial_duration",
        subset="(trial_start==True and onset > 0) & ~gain_amount.isnull()", demean_amp=True, 
        cond_id='trial_gain'
    )
    trial_loss = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="absolute_loss_amount", duration_column="trial_duration",
        subset="(trial_start==True and onset > 0) & ~absolute_loss_amount.isnull()", demean_amp=True, 
        cond_id='trial_loss'
    )
    design_matrix = pd.concat([all_task, pos_draw, neg_draw, trial_gain, 
        trial_loss, confound_regressors], axis=1)
    contrasts = {'task': 'task',
                'trial_loss': 'trial_loss',
                'trial_gain': 'trial_gain',
                'positive_draw': 'positive_draw',#
                'negative_draw': 'negative_draw'#
                }
    if regress_rt != 'no_rt':
        print('RT cannot be modeled for this task')
    return design_matrix, contrasts, percent_junk


def make_basic_stopsignal_desmat(events_file, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic stop signal regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'stopSignal')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and '
                            'omission == 0 and onset > 0')
    events_df['constant_1_column'] = 1  
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'too_fast'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'omission'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset='onset > 0', demean_amp = False, cond_id = 'commission'
        )
    go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'go'", 
        demean_amp=False, cond_id='go'
    )
    stop_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'stop_success'", 
        demean_amp=False, cond_id='stop_success'
    )
    stop_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'stop_failure'", 
        demean_amp=False, cond_id='stop_failure'
    )
    design_matrix = pd.concat([go, stop_success, stop_failure, too_fast_regressor, 
        omission_regressor, commission_regressor, confound_regressors], axis=1)
    contrasts = {#'go': 'go', #
                  #  'stop_success': 'stop_success',#
                  #  'stop_failure': 'stop_failure',#
                    'stop_success-go': 'stop_success-go',
                    'stop_failure-go': 'stop_failure-go',
                  #  'stop_success-stop_failure': 'stop_success-stop_failure',
                  #  'stop_failure-stop_success': 'stop_failure-stop_success',
                  #  'task': '.333*go + .333*stop_failure + .333*stop_success'#
                  }
    if regress_rt == 'rt_centered':
        rt_subset = subset_main_regressors + ' and trial_type != "stop_success"'
        mn_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt_subset = subset_main_regressors + ' and trial_type != "stop_success"'
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration':
        rt_subset = subset_main_regressors + ' and trial_type != "stop_success"'
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk


def make_basic_two_by_two_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic two by two regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requested.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'twoByTwo')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and '
                            'omission == 0 and onset > 0')
    events_df['constant_1_column'] = 1  
    events_df.trial_type = ['cue_'+c if c is not np.nan else 'task_'+t
                            for c, t in zip(events_df.cue_switch,
                                            events_df.task_switch)]
    events_df.trial_type.replace('cue_switch', 'task_stay_cue_switch',
                                inplace=True)

    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'too_fast'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'commission'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset="onset > 0", demean_amp = False, cond_id = 'omission'
        )
    task_switch_900 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .900 and trial_type == 'task_switch'", 
        demean_amp=False, cond_id='task_switch_900'
    )
    task_stay_cue_switch_900 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .900 and trial_type == 'task_stay_cue_switch'",
        demean_amp=False, cond_id='task_stay_cue_switch_900'
    )
    cue_stay_900 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .900 and trial_type == 'cue_stay'", 
        demean_amp=False, cond_id='cue_stay_900'
    )
    task_switch_100 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .100 and trial_type == 'task_switch'", 
        demean_amp=False, cond_id='task_switch_100'
    )
    task_stay_cue_switch_100 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .100 and trial_type == 'task_stay_cue_switch'", 
        demean_amp=False, cond_id='task_stay_cue_switch_100'
    )
    cue_stay_100 = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and CTI == .100 and trial_type == 'cue_stay'", 
        demean_amp=False, cond_id='cue_stay_100'
    )
    design_matrix = pd.concat([task_switch_900, task_stay_cue_switch_900, 
        cue_stay_900, task_switch_100, task_stay_cue_switch_100, cue_stay_100, 
        too_fast_regressor, commission_regressor, omission_regressor, confound_regressors], axis=1)
    contrasts = {'task_switch_cost_900': 'task_switch_900-task_stay_cue_switch_900',
                    'cue_switch_cost_900': 'task_stay_cue_switch_900-cue_stay_900',
                    'task_switch_cost_100': 'task_switch_100-task_stay_cue_switch_100',
                    'cue_switch_cost_100': 'task_stay_cue_switch_100-cue_stay_100'}
#                 #   'task_switch_cost': '(.5*task_switch_900+.5*task_switch_100)-'#
#                 #                       '(.5*task_stay_cue_switch_900+'
#                 #                       '.5*task_stay_cue_switch_100)', 
#                #    'cue_switch_cost': '(.5*task_stay_cue_switch_900+'#
#                #                       '.5*task_stay_cue_switch_100)'
#                #                       '-(.5*cue_stay_900+.5*cue_stay_100)', 
#                #    'task': '1/6*task_switch_900 + 1/6*task_switch_100 +' 
#                #            '1/6*task_stay_cue_switch_900 +'
#                #            ' 1/6*task_stay_cue_switch_100 + '
#                #            '1/6*cue_stay_900 + 1/6*cue_stay_100'
    if regress_rt == 'rt_centered':
        mn_rt = events_df.query(subset_main_regressors)['response_time'].mean()
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk


def make_basic_watt3_desmat(events_file, add_deriv, regress_rt, 
    n_scans, tr, confound_regressors
):
    """Creates basic WATT3 regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    #no junk trial definition for this task
    percent_junk = 0
    events_df['constant_1_column'] = 1  
    events_df['constant_3500ms_col'] = 3.5 * events_df['constant_1_column']
    events_df['constant_600ms_col'] = 0.6 * events_df['constant_1_column']
    events_df[['practice_main', 'with_without', 'not_using']] = \
        events_df.condition.str.split(expand=True, pat='_')
    events_df.with_without = events_df.with_without.replace('without', -1)
    events_df.with_without = events_df.with_without.replace('with', 1)
    events_df.block_duration = events_df.block_duration/1000
 
    planning_event = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_3500ms_col",
        subset="planning==1 and practice_main == 'PA'", 
        demean_amp=False, cond_id='planning_event'
    )
    planning_parametric = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="with_without", duration_column="constant_3500ms_col",
        subset="planning==1 and practice_main == 'PA'", 
        demean_amp=True, cond_id='planning_parametric'
    )
    acting_event = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_600ms_col",
        subset="planning==0 and practice_main == 'PA'", 
        demean_amp=False, cond_id='acting_event'
    )
    acting_parametric = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="with_without", duration_column="constant_600ms_col",
        subset="planning==0 and trial_id!='feedback' and practice_main == 'PA'", 
        demean_amp=True, cond_id='acting_parametric'
    )
    feedback = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="block_duration",
        subset="trial_id=='feedback' and practice_main == 'PA'", 
        demean_amp=False, cond_id='feedback'
    )
    # Practice starts here:
    acting_event_practice = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_600ms_col",
        subset="planning==0 and practice_main == 'UA'", 
        demean_amp=False, cond_id='acting_event_practice'
    )
    acting_parametric_practice = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="with_without", duration_column="constant_600ms_col",
        subset="planning==0 and trial_id!='feedback' and practice_main == 'UA'", 
        demean_amp=True, cond_id='acting_parametric_practice'
    )
    feedback_practice = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="block_duration",
        subset="trial_id=='feedback' and practice_main == 'UA'", demean_amp=False, 
        cond_id='feedback_practice'
    )
    design_matrix = pd.concat([planning_event, planning_parametric, acting_event, 
        acting_parametric, feedback, acting_event_practice, 
        acting_parametric_practice, feedback_practice, 
        confound_regressors], axis=1)
    contrasts = {'planning_event':'planning_event', 
                 'planning_parametric':'planning_parametric',
                 'acting_event':'acting_event', 
                 'acting_parametric': 'acting_parametric',
                 'feedback':'feedback', 
                 'task': '.5*planning_event + .5*acting_event',
                 'task_parametric': '.5*planning_parametric + .5*acting_parametric'
                 }
    if regress_rt != 'no_rt':
        print('RT cannot be modeled for this task')
    return design_matrix, contrasts, percent_junk


def make_basic_discount_fix_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
):
    """Creates basic discount fix regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'discountFix')
    #commission and omission are all 0s by definition
    subset_main_regressors = ('too_fast == 0 and key_press != -1')
    events_df['constant_1_column'] = 1  
    events_df['choice_parametric'] = -1
    events_df.loc[events_df.trial_type == 'larger_later',
                  'choice_parametric'] = 1

    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    task = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset='too_fast == 0', demean_amp=False, 
        cond_id='task'
    )
    choice = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="choice_parametric", duration_column="constant_1_column",
        subset='too_fast == 0', demean_amp=True, 
        cond_id='choice'
    )
    design_matrix = pd.concat([task, choice, too_fast_regressor, 
        confound_regressors], axis=1)
    contrasts = {'task': 'task',
                 'choice': 'choice'}
    if regress_rt == 'rt_centered':
        mn_rt = events_df.query(subset_main_regressors)['response_time'].mean()
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk


def make_basic_dpx_desmat(events_file, add_deriv, 
    regress_rt, n_scans, tr, confound_regressors
    ):
    """Creates basic dpx regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'DPX')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and '
                            'omission == 0')
    events_df['constant_1_column'] = 1  
    percent_junk = np.mean(events_df['too_fast'])
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'commission'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'omission'
        )
    AX = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'AX'", 
        demean_amp=False, cond_id='AX'
    )
    AY = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'AY'", 
        demean_amp=False, cond_id='AY'
    )
    BX = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'BX'", 
        demean_amp=False, cond_id='BX'
    )
    BY = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and condition == 'BY'", 
        demean_amp=False, cond_id='BY'
    )
    design_matrix = pd.concat([AX, AY, BX, BY, 
        too_fast_regressor, commission_regressor, omission_regressor, confound_regressors], axis=1)
    contrasts = {#'AX': 'AX',#
                 #'BX': 'BX',#
                 #'AY': 'AY',#
                 #'BY': 'BY',#
                 #'task': '.25*AX + .25*BX + .25*AY + .25*BY',
                 'AY-BY': 'AY-BY', 
                 'BX-BY': 'BX-BY'}
    if regress_rt == 'rt_centered':
        mn_rt = events_df.query(subset_main_regressors)['response_time'].mean()
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration':
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration_only':
        rt_AX = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors + " and condition == 'AX'", demean_amp=False, cond_id='ax_rtdur'
        ) 
        rt_AY = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors + " and condition == 'AY'", demean_amp=False, cond_id='ay_rtdur'
        )
        rt_BX = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors + " and condition == 'BX'", demean_amp=False, cond_id='bx_rtdur'
        ) 
        rt_BY = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=subset_main_regressors + " and condition == 'BY'", demean_amp=False, cond_id='by_rtdur'
        )
        design_matrix = pd.concat([rt_AX, rt_AY, rt_BX, rt_BY, 
        too_fast_regressor, commission_regressor, omission_regressor, confound_regressors], axis=1)
        contrasts = {'AY_rtdur-BY_rtdur': 'ay_rtdur-by_rtdur', 
                 'BX_rtdur-BY_rtdur': 'bx_rtdur-by_rtdur'}
    return design_matrix, contrasts, percent_junk


def make_basic_motor_selective_stop_desmat(events_file, add_deriv,
    regress_rt, n_scans, tr, confound_regressors
    ):
    """Creates basic Motor selective stop regressors (and derivatives)
       Input
         events_df: events data frame (events.tsv data)
         n_scans: Number of scans
         tr: time resolution
         confound_regressors: Confounds derived from fmriprep output
       Return
          design_matrix: pandas data frame including all regressors (and derivatives)
          contrasts: dictionary of contrasts in nilearn friendly format
          rt_subset: Boolean for extracting correct rows of events_df in the case that 
              rt regressors are requeset.
    """
    events_df = pd.read_csv(events_file, sep = '\t')
    events_df, percent_junk = define_nuisance_trials(events_df, 'motorSelectiveStop')
    subset_main_regressors = ('too_fast == 0 and commission == 0 and omission == 0')
    events_df['constant_1_column'] = 1  
    too_fast_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="too_fast", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'too_fast'
        )
    commission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="commission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'commission'
        )
    omission_regressor = make_regressor_and_derivative(
            n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
            amplitude_column="omission", duration_column="constant_1_column",
            subset=None, demean_amp = False, cond_id = 'omission'
        )
    crit_go = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_go'", 
        demean_amp=False, cond_id='crit_go'
    )
    crit_stop_success = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_stop_success'",
        demean_amp=False, cond_id='crit_stop_success'
    )
    crit_stop_failure = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'crit_stop_failure'", 
        demean_amp=False, cond_id='crit_stop_failure'
    )
    noncrit_signal = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'noncrit_signal'", 
        demean_amp=False, cond_id='noncrit_signal'
    )
    noncrit_nosignal = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="constant_1_column",
        subset=subset_main_regressors + " and trial_type == 'noncrit_nosignal'", 
        demean_amp=False, cond_id='noncrit_nosignal'
    )
    design_matrix = pd.concat([crit_go, crit_stop_success, crit_stop_failure,
        noncrit_signal, noncrit_nosignal,too_fast_regressor, 
        commission_regressor, omission_regressor, confound_regressors], axis=1)
    contrasts = {#'crit_go': 'crit_go',#
                 #'crit_stop_success': 'crit_stop_success',#
                 #'crit_stop_failure': 'crit_stop_failure',#
                 #'noncrit_signal': 'noncrit_signal',#
                 #'noncrit_nosignal': 'noncrit_nosignal',#
                 #'crit_stop_success-crit_go': 'crit_stop_success-crit_go', #
                 'crit_stop_failure-crit_go': 'crit_stop_failure-crit_go', 
                 #'crit_stop_success-crit_stop_failure': 'crit_stop_success-crit_stop_failure',#
                 #'crit_go-noncrit_nosignal': 'crit_go-noncrit_nosignal',#
                 #'noncrit_signal-noncrit_nosignal': 'noncrit_signal-noncrit_nosignal',#
                 #'crit_stop_success-noncrit_signal': 'crit_stop_success-noncrit_signal',
                 'crit_stop_failure-noncrit_signal': 'crit_stop_failure-noncrit_signal',
                 #'task': '.2*crit_go + .2*crit_stop_success +'#
                 #        '.2*crit_stop_failure + .2*noncrit_signal + .2*noncrit_nosignal'
                 }
    if regress_rt == 'rt_centered':
        rt_subset = subset_main_regressors + " and trial_type!='crit_stop_success'"
        mn_rt = events_df.query(rt_subset)['response_time'].mean()
        events_df['response_time_centered'] = events_df.response_time - mn_rt
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time_centered", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_uncentered':
        rt_subset = subset_main_regressors + " and trial_type!='crit_stop_success'"
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="response_time", duration_column="constant_1_column",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    if regress_rt == 'rt_duration':
        rt_subset = subset_main_regressors + " and trial_type!='crit_stop_success'"
        rt = make_regressor_and_derivative(
        n_scans=n_scans, tr=tr, events_df=events_df, add_deriv = add_deriv,
        amplitude_column="constant_1_column", duration_column="response_time",
        subset=rt_subset, demean_amp=False, cond_id='response_time'
        ) 
        design_matrix = pd.concat([design_matrix, rt], axis=1)
        contrasts["response_time"] = "response_time"
    return design_matrix, contrasts, percent_junk


make_task_desmat_fcn_dict = {
        'stroop': make_basic_stroop_desmat,
        'ANT': make_basic_ant_desmat,
        'CCTHot': make_basic_ccthot_desmat,
        'stopSignal': make_basic_stopsignal_desmat,
        'twoByTwo': make_basic_two_by_two_desmat,
        'WATT3': make_basic_watt3_desmat,
        'discountFix': make_basic_discount_fix_desmat,
        'DPX': make_basic_dpx_desmat,
        'motorSelectiveStop': make_basic_motor_selective_stop_desmat
    }

