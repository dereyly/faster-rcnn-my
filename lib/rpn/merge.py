# --------------------------------------------------------
# Hello PyCaffe
# Data Layer for Scale Augmantation
# LGPL
# Written by Nikolay Sergievsky (dereyly)
# --------------------------------------------------------
import sys
# sys.path.append('/home/dereyly/progs/caffe_cudnn33/python')
# sys.path.append('/usr/lib/python2.7/dist-packages')
# sys.path.insert(0,'/home/dereyly/progs/caffe-master-triplet/python')
# sys.path.insert(0,'/home/dereyly/progs/caffe-elu/python')

import caffe
import numpy as np
import yaml
import scipy.io as sio
import yaml
from caffe.proto import caffe_pb2
import lmdb
import cPickle as pkl
import random
import cv2
from google.protobuf import text_format


'''
layer {
  name: "merge_labels2"
  type: "Python"
  bottom: "labels1"
  bottom: "labels2"
  top: "labels3"
  python_param {
    module: 'rpn.merge'
    layer: 'Merge'
    }
}

'''

class Merge(caffe.Layer):

    def setup(self, bottom, top):
        #sz = bottom[0].data.shape
        #for i in range(len(bottom)):
            #sz=bottom[i].data.shape
        top[0].reshape(*bottom[0].data.shape)


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):

        merged = bottom[0].data
        for i in range(1,len(bottom)):
            merged=np.vstack((merged,bottom[i].data))
        top[0].reshape(*merged.shape)
        top[0].data[...]=merged
        pass


    def backward(self, top, propagate_down, bottom):
        # print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        pass

