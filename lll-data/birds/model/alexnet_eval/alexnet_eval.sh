#!/bin/bash
./build/tools/caffe train \
    -solver data/birds/model/alexnet_eval/alexnet_eval_solver.prototxt \
    -weights data/birds/model/bvlc_reference_caffenet.caffemodel \
    -gpu 0 2>&1 | tee data/birds/model/alexnet_eval/birds_alexnet_eval_log.txt
