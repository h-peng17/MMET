#! /bin/bash

if [ $1 == "com_img_rep" ] 
then
    echo $1
    python3 recall.py \
        --data_path $2 \
        --img_save_name $3 \
        --mode com_img_rep 
elif [ $1 == "con_img_lib" ]
then 
    echo $1
    python3 recall.py  \
    --img_save_name library_vector \
    --mode con_img_lib 
elif [ $1 == "con_text_lib" ] 
then
    echo $1
    python3 recall.py \
    --data_path ../data/library_text.json \
    --mode con_text_lib 
elif [ $1 == "query_text" ]
then 
    echo $1
    python3 recall.py  \
    --prefix $2 \
    --data_path "../data/$2data.json" \
    --mode query_text 
elif [ $1 == "query_image" ]
then 
    echo $1
    python3 recall.py \
    --prefix $2 \
    --img_save_name "$2vector" \
    --mode query_image
fi 

