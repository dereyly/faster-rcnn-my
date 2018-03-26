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
# import lmdb
import cPickle as pkl
import random
import cv2
from google.protobuf import text_format


'''
layer {
  name: "catcher"
  type: "Python"
  bottom: "res3b3_relu_mbox_priorbox"
  python_param {
    module: 'catcher'
    layer: 'Catcher'
    }
}

'''

class Catcher(caffe.Layer):

    def setup(self, bottom, top):
        #layer_params = yaml.load(self.param_str)
       # str_input_dim = layer_params['rsz_dim']
        #data = np.array(str_input_dim.split(' '), dtype='|S4')
        #self.rsz_dim=int(str_input_dim)
        sz = bottom[0].data.shape
        sz1 = bottom[1].data.shape
        sz2 = bottom[2].data.shape
        sz3 = bottom[3].data.shape
        #top[0].reshape(sz[0], sz[1], self.rsz_dim, self.rsz_dim)
        zz=0
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        sz = bottom[0].data.shape
        # sz1 = bottom[1].data.shape
        # sz2 = bottom[2].data.shape
        data=bottom[0].data
        #rects=bottom[1].data[0]
        #data1=bottom[1].data
        #sz2=bottom[1].data.shape
        #f_pkl = open('/home/dereyly/progs/caffe-ssd/priorbox_up.pkl', 'w')
        #pkl.dump(data,f_pkl)
        #f_pkl.close()
        pass


    def backward(self, top, propagate_down, bottom):
        # print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        pass

