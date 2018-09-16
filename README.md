# ECCV18_Lifelong_Learning
Thi repository is for the paper "Lifelong Learning via Progressive Distillation and Retrospection".
# Instructions
1. Install the dependencies for Caffe according to the official instructions and modify the ``./lll-caffe/Makefile.config``.
2. Compile the ``./lll-caffe`` and make a soft link to the ``./lll-data``.
```
cd lll-caffe
make -j  12
ln -s ../lll-data data
```
3. Download the pretrained models from the following links and put them in the ``./lll-models``.
[Link1](http://rec.ustc.edu.cn/s/jma3ej) (Password: [wcdfwu](wcdfwu))
4. Generate the lmdb for each dataset in the ``./lll-data`` according to the provided image list.
```
#Take Flowers as an example
./lll-data/data/flowers_train_lmdb #train set
./lll-data/data/flowers_test_lmdb #test set
./lll-data/data/flowers_seed1_uniform5_train_lmdb #subset from train set for Retrospection
```

5. ``ImageNet2Flowers`` is taken as an example to illustrate the usage of the code. Please refer to ``imagenet2flowers.sh`` for the details, including training and evaluation.
