import _init_paths
from fast_rcnn.test import *
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
# from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import sys
# sys.path.insert(0,'/home/dereyly/progs/group_caffe/caffe-fast-rcnn/python')
#sys.path.insert(0,'/home/dereyly/progs/py-RFCN-priv/caffe-priv/python')
sys.path.insert(0,'/home/dereyly/progs/caffe-nccl/python')
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms, soft_nms
import cPickle
from utils.blob import im_list_to_blob
import os
# os.chdir('..')
current_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_path +'/..')
#--def=/home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-v2-merge.prototxt --net=/home/dereyly/progs/py-RFCN-priv/output/faster_rcnn_end2end/voc_2007_trainval/faster_voc_resnet101-v2_ss_iter_100000.caffemodel
#--gpu 1  --def models/pvanet/example_train/test_5.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_5c_iter_130000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#--gpu 0 --def /home/dereyly/progs/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-food.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_5180_trainval/res101_food_iter_25000.caffemodel --cfg models/pvanet/cfgs/submit_food.yml
#--gpu 0 --def /home/dereyly/progs/pva-faster-rcnn/models/20170821_dairy_from_20170818__v5_3_123/models/test_food_ctg.prototxt --net /home/dereyly/progs/pva-faster-rcnn/models/20170821_dairy_from_20170818__v5_3_123/models/pvanet_frcnn_384_dairy_20170818_from_20170818_v5_3_123_after_45k_iter_15000.caffemodel --cfg models/pvanet/cfgs/submit_food.yml
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/caffe-model/faster_rcnn_end2end.yml', type=str)
    parser.add_argument('--fname_tst', dest='fname_tst',
                        help='name of file with paths to imgs',
                        default='/home/dereyly/ImageDB/food/VOC5180/ImageSets/Main/test.txt', type=str)
                        #default='/home/dereyly/ImageDB/VOCPascal/VOCdevkit/VOC2007/ImageSets/Main/test.txt', type=str)
    parser.add_argument('--img_dir', dest='img_dir',
                        default='/home/dereyly/ImageDB/food/VOC5180/JPEGImages/', type=str)
                        #default='/home/dereyly/ImageDB/VOCPascal/VOCdevkit/VOC2007/JPEGImages/', type=str)
    parser.add_argument('--out_dir', dest='out_dir',
                        default='/home/dereyly/ImageDB/food/VOC5180/res_pva1/', type=str)
                        #default = '/home/dereyly/ImageDB/VOCPascal/VOCdevkit/VOC2007/res/', type = str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def test_simple(net, fname_tst, img_dir, out_dir, max_per_image=400, thresh=-np.inf, vis=False):
    """Test a Fast R-CNN network on an image database."""
    is_eval=True
    save_count=100
    num_classes=20
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    data= {}


    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    count=0
    files=[]
    with open(fname_tst, 'r') as f_in:
        for line in f_in.readlines():
            files.append(line.strip())

    for count in range(len(files)):
            #count+=1
            line=files[count]
            # filter out any ground truth boxes
            path = img_dir+line +'.jpg'
            im = cv2.imread(path)
            _t['im_detect'].tic()
            scores, boxes = im_detect(net, im)
            _t['im_detect'].toc()

            _t['misc'].tic()
            # skip j = 0, because it's the background class
            dets_all= np.empty((0,6), np.float32)
            scores_all=np.empty((0,num_classes), np.float32)
            for j in xrange(1, num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[inds, j]

                cls_boxes = boxes[inds, j*4:(j+1)*4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis],)) \
                    .astype(np.float32, copy=False)
                #keep = soft_nms(cls_dets, method=1)
                #keep = nms(dets_all[:, :5].astype(np.float32), cfg.TEST.NMS)
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = np.hstack((cls_dets, j * np.ones((cls_dets.shape[0], 1))))
                if len(cls_dets) > 0:
                    dets_all = np.vstack((dets_all, cls_dets[keep].copy()))
                    scores_all = np.vstack((scores_all, scores[inds[keep]].copy()))

            #keep = nms(dets_all[:,:5].astype(np.float32), cfg.TEST.NMS)
            #keep = soft_nms(cls_dets, method=1)
            # dets_all=dets_all[keep]
            #data[line] = { 'bbox':dets_all[keep].copy(),'cls':scores_all[keep].copy()}
            data[line] = {'bbox': dets_all.copy(), 'cls': scores_all.copy()}

            _t['misc'].toc()
            print(count,line, _t['im_detect'].average_time,_t['misc'].average_time)
            if count+1 % save_count==0 or count+1==len(files):
                det_file = os.path.join(out_dir, 'det_%d.pkl' %count)
                with open(det_file, 'wb') as f:
                    cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
                data={}




if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    test_simple(net, args.fname_tst, args.img_dir, args.out_dir, max_per_image=400)
