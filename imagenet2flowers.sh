#Prepare
#bvlc_reference_caffenet.caffemodel is pretrained on ImageNet
cd lll-data/data/flowers/model
ln -s ../../../lll-models/bvlc_reference_caffenet.caffemodel bvlc_reference_caffenet.caffemodel
ln -s bvlc_reference_caffenet.caffemodel bvlc_reference_caffenet.caffemodel_+task1
ln -s bvlc_reference_caffenet.caffemodel bvlc_reference_caffenet.caffemodel_+task2
ln -s bvlc_reference_caffenet.caffemodel imagenet_alexnet_train_iter_15000.caffemodel #for eval
#flowers_alexnet_train_iter_15000.caffemodel is finetuned on flowers from the model pretrained on ImageNet
ln -s ../../../lll-models/flowers_alexnet_train_iter_15000.caffemodel flowers_alexnet_train_iter_15000.caffemodel 
ln -s flowers_alexnet_train_iter_15000.caffemodel flowers_alexnet_train_iter_15000.caffemodel_+task2ft

#Train
cd lll-caffe
#1. Learn without Forgetting (LwF)
sh data/flowers/model/alexnet_KD_from_imagenet/v2_alexnet_KD_from_imagenet.sh $GPUID
#2. Distillation (D)
sh data/flowers/model/alexnetSL_KD_from_imagenet/v2_alexnetSL_KD_from_imagenet.sh $GPUID
#3. LwF+Retrospection (R)
sh data/flowers/model/alexnetPDA_KD_from_imagenet/v2_alexnetPDA_KD_from_imagenet.sh $GPUID
#4. Distillation+Retrospection (D+R)
sh data/flowers/model/SLPDA_alexnet_KD_from_imagenet/v2_SLPDA_alexnet_KD_from_imagenet.sh $GPUID

#Eval
cd lll-caffe/data/eval
#1. Learn without Forgetting (LwF)
./lll_eval.sh KDimagenet2flowers flowers model alexnet_KD_from_imagenet v2_alexnet_KD_from_imagenet 15000 0 #on Flowers
./lll_eval.sh KDimagenet2flowers flowers model alexnet_KD_from_imagenet v2_alexnet_KD_from_imagenet 15000 0 #on ImageNet
#2. Distillation (D)
./lll_eval.sh KDimagenet2flowers flowers model alexnetSL_KD_from_imagenet v2_alexnetSL_KD_from_imagenet 15000 0 #on Flowers
./lll_eval.sh KDimagenet2flowers flowers model alexnetSL_KD_from_imagenet v2_alexnetSL_KD_from_imagenet 15000 0 #on ImageNet
#3. LwF+Retrospection (R)
./lll_eval.sh KDimagenet2flowers flowers model alexnetPDA_KD_from_imagenet v2_alexnetPDA_KD_from_imagenet 15000 0 #on Flowers
./lll_eval.sh KDimagenet2flowers flowers model alexnetPDA_KD_from_imagenet v2_alexnetPDA_KD_from_imagenet 15000 0 #on ImageNet
#4. Distillation+Retrospection (D+R)
./lll_eval.sh KDimagenet2flowers flowers model SLPDA_alexnet_KD_from_imagenet v2_SLPDA_alexnet_KD_from_imagenet 15000 0 #on Flowers
./lll_eval.sh KDimagenet2flowers flowers model SLPDA_alexnet_KD_from_imagenet v2_SLPDA_alexnet_KD_from_imagenet 15000 0 #on ImageNet