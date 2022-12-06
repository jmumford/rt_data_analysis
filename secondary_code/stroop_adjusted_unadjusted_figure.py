from nilearn.image import threshold_img, math_img, binarize_img
from nilearn.plotting import plot_roi
from matplotlib import colors
import matplotlib.pyplot as plt

stroop_lev2 = '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/stroop_lev2_output'
stroop_no_lev1_rt = (f'{stroop_lev2}/stroop_lev1_contrast_stroop_incong_minus_'
    'cong_rtmod_no_rt_duration_constant_lev2_model_one_sampt/'
    'mean_diff0_2sided_tfce_corrp_fstat1.nii.gz'
)
stroop_yes_lev1_rt = (f'{stroop_lev2}/stroop_lev1_contrast_stroop_incong_minus_'
    'cong_rtmod_rt_uncentered_duration_constant_lev2_model_one_sampt/'
    'mean_diff0_2sided_tfce_corrp_fstat1.nii.gz'
)
no_lev1_rt_thresh = binarize_img(threshold_img(stroop_no_lev1_rt, .95))
yes_lev1_rt_thresh = binarize_img(threshold_img(stroop_yes_lev1_rt, .95))
no_only1_yes_only2_overlap3 = math_img('img1 + 2*img2', 
    img1 = no_lev1_rt_thresh, img2 = yes_lev1_rt_thresh,
)

cmap_mine = colors.ListedColormap(['darkorange', 'magenta', 'dodgerblue'])
plot_roi(no_only1_yes_only2_overlap3, alpha=1, view_type='continuous', 
    display_mode='z', cut_coords = 9, cmap=cmap_mine)
plt.savefig('/oak/stanford/groups/russpold/data/uh2/aim1_mumford/figures/'
    'stroop_mean_incon_vs_cong_w_rt_blue_wout_rt_orange.pdf')