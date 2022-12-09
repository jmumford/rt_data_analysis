import glob
from nilearn.image import binarize_img, concat_imgs, mean_img, threshold_img, math_img
from nilearn.plotting import plot_roi, plot_img
from nilearn.datasets import load_mni152_template
from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


#Conjunction map for 7 tasks
outdir = '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output'
#This assumes lev2 directories only exist for 7 tasks of interest 
# (make sure ccthot and WATT3 are omitted)
files_for_conjunction = glob.glob(
    f'{outdir}/*lev2*/*contrast_response_time*one_sampt/'
    '*corrp_fstat*'
)

dat_4d = concat_imgs(files_for_conjunction)
dat_4d_bin = binarize_img(dat_4d, threshold = 0.95)
dat_bin_avg = mean_img(dat_4d_bin)
dat_conj = binarize_img(dat_bin_avg, threshold = 1)

cmap_orange = colors.ListedColormap(['darkorange'])
plot_roi(dat_conj, alpha=1, view_type='continuous', 
    display_mode='z', cut_coords = 9, cmap=cmap_orange)
plt.savefig('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/figures/'
    'conjunction_avg_rt_effect_across_7tasks.pdf')

dat_conj.to_filename(
    '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/'
    'conjunction_output/rt_conjunction_new.nii.gz'
)

########
# Stroop effect (group level)  With and without trial-specific RT adjustment
# I didn't end up adding this to the paper since I didn't want to get into interpreting Stroop 
# and I didn't want to include all tasks with/without RT adjustment, since it seems unwieldy
 

stroop_lev2 = '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/stroop_lev2_output'
stroop_no_lev1_rt = (f'{stroop_lev2}/stroop_lev1_contrast_stroop_incong_minus_'
    'cong_rtmod_no_rt_lev2_model_one_sampt/'
    'randomise_output_model_one_sampt_tfce_corrp_fstat1.nii.gz'
)
stroop_yes_lev1_rt = (f'{stroop_lev2}/stroop_lev1_contrast_stroop_incong_minus_'
    'cong_rtmod_rt_uncentered_lev2_model_one_sampt/'
    'randomise_output_model_one_sampt_tfce_corrp_fstat1.nii.gz'
)
no_lev1_rt_thresh = binarize_img(threshold_img(stroop_no_lev1_rt, .95))
yes_lev1_rt_thresh = binarize_img(threshold_img(stroop_yes_lev1_rt, .95))
no_only1_yes_only2_overlap3 = math_img('img1 + 2*img2', 
    img1 = no_lev1_rt_thresh, img2 = yes_lev1_rt_thresh,
)

cmap_mine = colors.ListedColormap(['darkorange',  'dodgerblue', 'forestgreen'])
plot_roi(no_only1_yes_only2_overlap3, alpha=1, view_type='continuous', 
    display_mode='z', cut_coords = 9, cmap=cmap_mine)
plt.savefig('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/figures/'
    'stroop_mean_incon_vs_cong_w_rt_blue_wout_rt_orange_overlap_green.pdf')


########
# No longer used.  Previous result likely driven by outlier who was removed after
# Behavioral QA was incorporated.
# Correlation of Stroop effect (incon - con) with RT difference (incon -con)
# with and without between-trial RT adjustment

stroop_cor_w_rt_diff_nort_fp = (f'{outdir}/stroop_lev2_output/'
    'stroop_lev1_contrast_stroop_incong_minus_cong_rtmod_no_rt_'
    'lev2_model_rt_diff/randomise_output_model_rt_diff_tfce_corrp_fstat2.nii.gz')
stroop_cor_w_rt_diff_yesrt_fp = (f'{outdir}/stroop_lev2_output/'
    'stroop_lev1_contrast_stroop_incong_minus_cong_rtmod_rt_uncentered_'
    'lev2_model_rt_diff/'
    'randomise_output_model_rt_diff_tfce_corrp_fstat2.nii.gz')

stroop_cor_w_rt_diff_nort_p_thresh = binarize_img(
    threshold_img(stroop_cor_w_rt_diff_nort_fp, .95))
stroop_cor_w_rt_diff_yesrt_p_thresh = binarize_img(
    threshold_img(stroop_cor_w_rt_diff_yesrt_fp, .95))
no_only1_yes_only2_overlap3 = math_img('img1 + 2*img2', 
    img1 = stroop_cor_w_rt_diff_nort_p_thresh, 
    img2 = stroop_cor_w_rt_diff_yesrt_p_thresh
)

no_only1_yes_only2_overlap3_array = no_only1_yes_only2_overlap3.get_fdata()
print(np.unique(no_only1_yes_only2_overlap3_array, return_counts=True))
#Neither map has significant results, so not including this figure

# plot thresholded t-stat (f-test is 2-sided version of this t-stat)
stroop_cor_w_rt_diff_nort_t = (f'{outdir}/stroop_lev2_output/'
    'stroop_lev1_contrast_stroop_incong_minus_cong_rtmod_no_rt_duration_'
    'constant_lev2_model_rt_diff/mean_diff0_2sided_tstat2.nii.gz')
stroop_cor_w_rt_diff_nort_t_above_thresh = math_img('img1 * img2',
    img1=stroop_cor_w_rt_diff_nort_t, img2=stroop_cor_w_rt_diff_nort_p_thresh
)

stroop_cor_w_rt_diff_nort_t_above_thresh_array = stroop_cor_w_rt_diff_nort_t_above_thresh.get_fdata()
min_t_above_thresh = np.min(
    stroop_cor_w_rt_diff_nort_t_above_thresh_array
    [stroop_cor_w_rt_diff_nort_t_above_thresh_array>0])
np.max(
    stroop_cor_w_rt_diff_nort_t_above_thresh_array
    [stroop_cor_w_rt_diff_nort_t_above_thresh_array>0])

mni152template = load_mni152_template()
#Calculated a vmin and vmax for MNI so the background would look like my 
# plot_stat_roi() images.  That function didn't work here with a colorbar
# range that I liked (maybe because there are only positive values?)
plot_img(stroop_cor_w_rt_diff_nort_t_above_thresh,  
    display_mode='z', cut_coords = 9, threshold=min_t_above_thresh,
    colorbar=True, bg_img=mni152template, bg_vmin=-.4447, bg_vmax = .9882,
    cmap=mpl.colormaps['hot'])
plt.savefig('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/figures/'
    'stroop_incon_vs_cong_cor_w_rtdiff_nortadj_t_thresh_2sided.pdf')
