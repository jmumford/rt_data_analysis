import glob
from nilearn.image import binarize_img, concat_imgs, mean_img


# Conjunction map for all tasks
outdir = '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output'
files_for_conjunction = glob.glob(
    f'{outdir}/*lev2*/*contrast_response_time*one_sampt/'
    'mean_diff0_2sided_tfce_corrp_fstat1.nii.gz'
)

dat_4d = concat_imgs(files_for_conjunction)
dat_4d_bin = binarize_img(dat_4d, threshold = 0.95)
dat_bin_avg = mean_img(dat_4d_bin)
dat_conj = binarize_img(dat_bin_avg, threshold = 1)

dat_conj.to_filename(
    '/oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/'
    'conjunction_output/rt_conjunction.nii.gz'
)

