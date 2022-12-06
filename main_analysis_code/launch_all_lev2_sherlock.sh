#!/usr/bin/bash

#Use with caution, as this will run all analyses, even if they've been run before
all_batch=$(ls /oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/*lev2_output/*model_one_sampt/*batch)

for cur_batch in ${all_batch}
do
  sbatch ${cur_batch}
done

