#!/bin/bash
./build/tools/caffe train \
    -solver data/mitindoor/model/alexnet_eval/alexnet_eval_solver.prototxt \
    -weights data/mitindoor/model/bvlc_reference_caffenet.caffemodel \
    -gpu 0 2>&1 | tee data/mitindoor/model/alexnet_eval/mitindoor_alexnet_eval_log.txt
