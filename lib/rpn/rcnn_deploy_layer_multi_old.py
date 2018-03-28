# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
'''
layer {
  name: 'rcnn-ancors'
  type: 'Python'
  bottom: 'rois'
  bottom: 'multi_rcnn_probs'
  top: 'cls_prob'
  python_param {
    module: 'rpn.rcnn_deploy_layer'
    layer: 'RCNNDeployLayer'
    param_str: "{'ratios': [0.5, 1, 2], 'scales': [2, 4, 8, 16, 32, 64]}"
  }
}
'''
import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
from generate_anchors import generate_anchors
DEBUG = False

class RCNNDeployLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        anchor_ratios = layer_params.get('ratios', ((0.5, 1, 2)))
        _base_size = layer_params.get('base_size', 16)
        self.num_cls=21

        self._anchors = generate_anchors(base_size=_base_size, ratios=anchor_ratios, scales=np.array(anchor_scales))
        x_ctr = (_base_size-1.0) / 2
        self._anchors-=x_ctr
        self._num_anchors = self._anchors.shape[0]
        self.th_ov=1.5
        top[0].reshape(1)
        top[1].reshape(1)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        probs=[]
        for i in range(self._num_anchors):
            probs.append(bottom[1+i].data.copy())
        probs = np.array(probs)

        bbox_pred = []
        for i in range(self._num_anchors):
            bbox_pred.append(bottom[1+self._num_anchors + i].data.copy())
        bbox_pred = np.array(bbox_pred)
        batch_size=len(all_rois)
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before

        try:
            all_rois=np.squeeze(all_rois,axis=(2,3))
        except:
            pass
        wh = all_rois[:, 3:5]-all_rois[:, 1:3]
        centred_rois=np.hstack((-wh/2,wh/2))
        overlaps = bbox_overlaps(
            np.ascontiguousarray(centred_rois, dtype=np.float),
            np.ascontiguousarray(self._anchors, dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)

        idx_ov_bool=overlaps>self.th_ov
        for k,id in enumerate(gt_assignment):
            idx_ov_bool[k,id]=True

        #idx_ov=np.where(overlaps>self.th_ov)
        probs_out = -np.ones((batch_size,self.num_cls))
        bbox_out = -np.ones((batch_size, self.num_cls*4))
        dbg_idx_ov =[]
        for k in range(all_rois.shape[0]):
            idx=np.where(idx_ov_bool[k])[0]
            dbg_idx_ov.append(idx.tolist())
            #probs_out[k] = np.max(probs[idx,k,:],axis=0)
            probs_out[k] = np.max(probs[idx, k, :], axis=0)
            bbox_out[k] = 0 #np.mean(bbox_pred[idx, k, :], axis=0)

        dbg=np.abs(probs[0] - probs[11]).sum(axis=1)
        ix_dbg=dbg.argmax()
        print (dbg>1).sum()

        #labels_out = np.expand_dims(labels_out,axis=1)

        zz=0
        #labels = labels[gt_assignment]


        top[0].reshape(*probs_out.shape)
        top[0].data[...] = probs_out
        top[1].reshape(*bbox_out.shape)
        top[1].data[...] = bbox_out
        zz=0

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



