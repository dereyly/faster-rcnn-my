__author__ = 'dereyly'
import sys
#sys.path.append('/home/dereyly/progs/caffe_cudnn33/python_33')
#sys.path.append('/home/dereyly/progs/caffe-master-triplet/python')
import caffe
import numpy as np

'''
layer {
  name: 'rcls_lost_my'
  type: 'Python'
  bottom: 'feats'
  bottom: 'labels'
  top: 'cls_lost_my'
  python_param {
    module: 'fast_rcnn.softmax_loss'
    layer: 'SoftmaxLossLayer'
    #param_str: "{'ratios': [0.5, 1, 2], 'scales': [2, 4, 8, 16, 32]}"
  }
}
'''

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sum_sf=np.sum(sf, axis=1)
    for i in range(x.shape[0]):
        sf[i]/=sum_sf[i]
    return sf

class SoftmaxLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        # difference is shape of inputs
        sz=bottom[0].data.shape
        self.batch_sz=sz[0]
        self.diff = np.zeros((sz[0],sz[1]),dtype=np.float32)
        self.lbl_gt=np.zeros((sz[0],sz[1]),dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        #top[1].reshape(self.batch_sz)

    def forward(self, bottom, top):
        sz=bottom[0].data.shape
        self.lbl_gt=np.zeros((sz[0],sz[1]),dtype=np.float32)
        lbl_idx=bottom[1].data
        lbl_idx=lbl_idx.astype(dtype= int)
        for i in range(self.batch_sz):
            self.lbl_gt[i,lbl_idx[i]]=1
        soft_max=softmax(bottom[0].data)
        #loss = -self.lbl_gt*np.log(np.maximum(soft_max,np.finfo(np.float32).eps))

        loss=0
        for i in range(self.batch_sz):
            loss -= np.log(np.maximum(soft_max[i][lbl_idx[i]],np.finfo(np.float32).eps))

        #loss2=-np.log(soft_max)
        #for i in range(self.batch_sz):
        #    loss[i,lbl_idx[i]]=0
        #print bottom[1].data.shape
        self.diff[...] = soft_max-self.lbl_gt

        top[0].data[...] = np.sum(loss) / bottom[0].num
        #top[1].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        #pass
        bottom[0].diff[...] = self.diff / bottom[0].num

