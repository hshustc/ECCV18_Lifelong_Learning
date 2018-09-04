#!/bin/bash
./build/tools/caffe train \
    -solver data/aircrafts/model/alexnet_eval/alexnet_eval_solver.prototxt \
    -weights data/aircrafts/model/bvlc_reference_caffenet.caffemodel \
    -gpu 0 2>&1 | tee data/aircrafts/model/alexnet_eval/aircrafts_alexnet_eval_log.txt
