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
  name: "MultyROI"
  type: "Python"
  bottom: "rois"
  bottom: "im_info"
  top: "rois2"
  top: "rois3"
  python_param {
    module: 'multy_roi'
    layer: 'MultyROI'
    }
}

'''

class MultyROI(caffe.Layer):

    def setup(self, bottom, top):

        sz = bottom[0].data.shape
        #self.multy_coef=[1.4, 1.8]
        self.multy_coef = [1.6]
        for i, K in enumerate(self.multy_coef):
            top[i].reshape(1)

        #top[2].reshape(3)
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        roi = bottom[0].data
        im_info = bottom[1].data[0, :]
        #roi_out=[]
        wh=roi[:,3:]-roi[:,1:3]
        for i,K in enumerate(self.multy_coef):
            roi_loc=np.copy(roi)
            roi_loc[:, 1:3] -= wh * ((K - 1) / 2)
            roi_loc[:, 3:] += wh * ((K - 1) / 2)
            roi_loc[np.where(roi_loc<0)]=0
            roi_loc[np.where(roi_loc[:, 2] >= im_info[1])] =im_info[1]-1
            roi_loc[np.where(roi_loc[:, 3] >= im_info[0])] = im_info[0] - 1
            top[i].reshape(*roi_loc.shape)
            top[i].data[...] = np.copy(roi_loc)
        pass


    def backward(self, top, propagate_down, bottom):
        # print 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
        pass

