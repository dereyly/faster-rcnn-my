#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
#/home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_iter_180000.caffemodel
#--gpu 0 --def models/pvanet/example_train/test.prototxt --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_iter_180000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#--gpu 0 --def models/pvanet/example_train/test_2.prototxt --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_v2_iter_160000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#/home/dereyly/progs/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-v2-merge.prototxt /home/dereyly/progs/pva-faster-rcnn/models/priv/faster_voc_resnet101-v2_ss_iter_100000.caffemodel
#--gpu 0 --def /home/dereyly/progs/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-v2-merge.prototxt --net /home/dereyly/progs/pva-faster-rcnn/models/priv/faster_voc_resnet101-v2_ss_iter_100000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#--gpu 0 --def /home/dereyly/progs/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-my.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/res101-my_iter_80000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml
#--gpu 0 --def models/pvanet/example_train/test_2.prototxt --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_v2_iter_160000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#--gpu 0 --def models/pvanet/example_train/test_multi-rcnn.prototxt --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_4_iter_180000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#--gpu 0 --def /home/dereyly/progs/pva-faster-rcnn/lib/generator/res101_multi_thread.prototxt --net /home/dereyly/progs/pva-faster-rcnn/models/multi_rcnn/faster_voc_resnet101-multi-anchrs_ss_iter_180000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import os
root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
os.chdir(root_path)
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
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    # while not os.path.exists(args.caffemodel) and args.wait:
    #     print('Waiting for {} to exist...'.format(args.caffemodel))
    #     time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net(net, imdb, max_per_image=args.max_per_image, vis=args.vis)
