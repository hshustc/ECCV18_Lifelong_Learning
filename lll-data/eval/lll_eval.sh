#!/bin/bash
if [ $# != 7 ]; then
    echo "./eval.sh TYPE DOMAIN MODEL_DIR MODEL MODEL_PREFIX ITER GPUID"
    echo "For example, ./eval.sh flowers2birds birds flowers_model alexnet alexnet 15000 0"
    exit 1
fi

TYPE=$1
DOMAIN=$2
MODEL_DIR=$3
MODEL=$4
MODEL_PREFIX=$5
ITER=$6
GPUID=$7

CAFFE_DIR=/home/housaihui/ECCV18_Lifelong_Learning/lll-caffe
OUT_DIR=$CAFFE_DIR/data/eval
CUR_DIR=`pwd`

if [[ $TYPE =~ "flowers2birds" ]]; then
        SOURCE=flowers
        TARGET=birds
elif [[ $TYPE =~ "imagenet2birds" ]]; then
        SOURCE=imagenet
        TARGET=birds
elif [[ $TYPE =~ "imagenet2flowers" ]]; then
        SOURCE=imagenet
        TARGET=flowers
elif [[ $TYPE =~ "imagenet2mitindoor" ]]; then
        SOURCE=imagenet
        TARGET=mitindoor
elif [[ $TYPE =~ "flowers2mitindoor" ]]; then
        SOURCE=flowers
        TARGET=mitindoor
elif [[ $TYPE =~ "flowers2aircrafts" ]]; then
        SOURCE=flowers
        TARGET=aircrafts
else
        echo "unknown TYPE"
        exit 2
fi

SOLVER=data/${DOMAIN}/model/alexnet_eval/alexnet_eval_solver.prototxt
if [[ "$TYPE" =~ "KD" ]]; then
    WEIGHTS0=${CAFFE_DIR}/data/${TARGET}/$MODEL_DIR/$MODEL/snapshot/${TARGET}_${MODEL_PREFIX}_train_iter_${ITER}.caffemodel
    WEIGHTS=${WEIGHTS0}_-task2
    if [ ! -f $WEIGHTS0 ]; then
        echo "$WEIGHTS0 does not exist"
        exit 4
    fi
    ln -s $WEIGHTS0 $WEIGHTS
else
    WEIGHTS0=${CAFFE_DIR}/data/${TARGET}/$MODEL_DIR/${DOMAIN}_alexnet_train_iter_15000.caffemodel #for fc8
    WEIGHTS1=${CAFFE_DIR}/data/${TARGET}/$MODEL_DIR/${MODEL}/snapshot/${TARGET}_${MODEL_PREFIX}_train_iter_${ITER}.caffemodel
    if [ ! -f $WEIGHTS0 ]; then
        echo "$WEIGHTS0 does not exist"
        exit 5
    fi
    if [ ! -f $WEIGHTS1 ]; then
        echo "$WEIGHTS1 does not exist"
        exit 6
    fi
    WEIGHTS="${WEIGHTS0},${WEIGHTS1}"
fi
#LOG=${CAFFE_DIR}/data/${TARGET}/$MODEL_DIR/$MODEL/${TYPE}_test_on_${DOMAIN}.log_${TARGET}_${MODEL_PREFIX}_train_iter_${ITER}.caffemodel

cd $CAFFE_DIR && ./build/tools/caffe train -solver $SOLVER -weights $WEIGHTS -gpu $GPUID
echo "#########################################################"
echo "cd $CAFFE_DIR && ./build/tools/caffe train -solver $SOLVER -weights $WEIGHTS -gpu $GPUID"
echo "#########################################################"
