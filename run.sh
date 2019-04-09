#!/bin/sh
python -u train.py \
    --breed cat \
    --model cat_scratch \
    --lr 0.003 \
    --momentum 0.9 \
    --weight-decay 0.5 \
    --batch-size 32 \
    --epochs 30 | tee cat_scratch.log