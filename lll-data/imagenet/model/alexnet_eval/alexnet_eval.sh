#!/bin/bash
./build/tools/caffe train \
    -solver data/imagenet/model/alexnet_eval/alexnet_eval_solver.prototxt \
    -weights data/imagenet/model/bvlc_reference_caffenet.caffemodel \
    -gpu 0 2>&1 | tee data/imagenet/model/alexnet_eval/imagenet_alexnet_eval_log.txt
