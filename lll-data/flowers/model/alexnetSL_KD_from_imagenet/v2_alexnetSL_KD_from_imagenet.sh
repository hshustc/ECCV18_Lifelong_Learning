#!/bin/bash
if [ $# != 1 ]; then
    echo "GPUID must be specified"
    exit 1
fi
GPUID=$1
./build/tools/caffe train \
    -solver data/flowers/model/alexnetSL_KD_from_imagenet/v2_alexnetSL_KD_from_imagenet_solver.prototxt \
    -weights data/flowers/model/flowers_alexnet_train_iter_15000.caffemodel_+task2ft,data/flowers/model/bvlc_reference_caffenet.caffemodel_+task1,data/flowers/model/bvlc_reference_caffenet.caffemodel_+task2 \
    -gpu $GPUID 2>&1 | tee data/flowers/model/alexnetSL_KD_from_imagenet/flowers_v2_alexnetSL_KD_from_imagenet_log.txt
