#!/bin/bash

inputs_dir=$1
save_dir=$2

if [[ $inputs_dir != */ ]]; then inputs_dir=$inputs_dir'/'; fi
if [[ $save_dir != */ ]]; then save_dir=$save_dir'/'; fi

mkdir $save_dir

for entry in "$inputs_dir"*
do  
    bname=$(basename $entry)
    echo "Process $bname"
    prefix=${bname%.*}
    mkdir $save_dir$prefix
    ./cameraMotionDesc -i $entry -w 4 -s $save_dir$prefix'/'$prefix

done