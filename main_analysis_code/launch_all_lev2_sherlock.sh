#!/usr/bin/bash

#Use with caution, as this will run all analyses, even if they've been run before
all_batch=$(ls /oak/stanford/groups/russpold/data/uh2/aim1_mumford/output/twoByTwo_lev2_output/*contrast*no_rt*/*batch)

for cur_batch in ${all_batch}
do
  sbatch ${cur_batch}
done

