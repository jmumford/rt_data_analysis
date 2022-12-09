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