#!/bin/sh
python -u train.py \
    --breed cat \
    --model cat_scratch \
    --lr 0.001 \
    --momentum 0.9 \
    --weight-decay 0.5 \
    --batch-size 26 \
    --epochs 20 | tee cat_scratch.log
