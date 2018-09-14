#!/bin/bash

CONTENT_DIR='/data/COCO14/content_train'
STYLE_DIR='/data/COCO14/style_train'
BATCH_SIZE=64
NPROC_PER_NODE=8
SAVE_EVERY_N_EPOCH=2

#echo $CONTENT_DIR
python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
        train.py \
        --content_dir=$CONTENT_DIR \
        --style_dir=$STYLE_DIR \
        --batch_size=$BATCH_SIZE \
        --save_every_n_epoch=$SAVE_EVERY_N_EPOCH
