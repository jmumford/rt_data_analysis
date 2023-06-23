#!/usr/bin/bash

all_batch=$(ls /oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/*lev1_output/batch_files/*)

for cur_batch in ${all_batch}
do
  sbatch ${cur_batch}
done

#all_batch=$(ls /oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/*lev1_output/batch_files/*rtmodel_no*)

#for cur_batch in ${all_batch}
#do
#  sbatch ${cur_batch}
#done
