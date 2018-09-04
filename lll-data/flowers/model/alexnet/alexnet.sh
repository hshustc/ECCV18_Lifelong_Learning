#!/bin/bash
./build/tools/caffe train \
    -solver data/flowers/model/alexnet/alexnet_solver.prototxt \
    -weights data/flowers/model/bvlc_reference_caffenet.caffemodel \
    -gpu 0 2>&1 | tee data/flowers/model/alexnet/flowers_alexnet_log.txt
