#!/usr/bin/env bash

src_root="./dense_neutral_exa"
tar_root="./dense_neutral_exa_exp"
while read model_name; do     # read txt file line by line
    # for .obj file
    echo "$model_name"
    tar_path=$tar_root"/shape/"$model_name
    echo "$tar_path"
    if [ ! -d "$tar_path" ]; then
        mkdir "$tar_path"
    fi
    src_path=$src_root"/"$model_name"/"
    echo "$src_path"
    find "$src_path" -name *.obj -exec mv -t "$tar_path" {} +

    # for .png file
    tar_path=$tar_root"/sketch/"$model_name
    if [ ! -d "$tar_path" ]; then
        mkdir "$tar_path"
    fi
    src_path=$src_root"/"$model_name"/"
    find "$src_path" -name *.png -exec mv -t "$tar_path" {} +
done < dense_neutral_list.txt