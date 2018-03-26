import _init_paths
# from fast_rcnn.test import *
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
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
                        default='/home/dereyly/ImageDB/food/VOC5180/res/', type=str)
                        #default = '/home/dereyly/ImageDB/VOCPascal/VOCdevkit/VOC2007/res/', type = str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        # Make width and height be multiples of a specified number
        im_scale_x = np.floor(im.shape[1] * im_scale / cfg.TEST.SCALE_MULTIPLE_OF) * cfg.TEST.SCALE_MULTIPLE_OF / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / cfg.TEST.SCALE_MULTIPLE_OF) * cfg.TEST.SCALE_MULTIPLE_OF / im.shape[0]
        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y]))
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, _t=None, boxes=None):

    if _t:
        _t['im_preproc'].tic()
    blobs, im_scales = _get_blobs(im, boxes)





    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [np.hstack((im_blob.shape[2], im_blob.shape[3], im_scales[0]))],
        dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))


    # do forward
    net.blobs['data'].data[...] = blobs['data']
    #forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

    net.blobs['im_info'].data[...] = blobs['im_info']

    if _t:
        _t['im_preproc'].toc()

    if _t:
        _t['im_net'].tic()
    blobs_out = net.forward()
    if _t:
        _t['im_net'].toc()
    #blobs_out = net.forward(**forward_kwargs)

    if _t:
        _t['im_postproc'].tic()

    assert len(im_scales) == 1, "Only single-image batch implemented"
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]


    scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if _t:
        _t['im_postproc'].toc()

    return scores, pred_boxes

def test_simple(net, fname_tst, img_dir, out_dir, max_per_image=400, thresh=-np.inf, vis=False):
    """Test a Fast R-CNN network on an image database."""
    is_eval=False
    save_count=100
    num_classes=23
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    data= {}
    if is_eval:
        imdb = get_imdb('voc_2007_test')
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    count=0
    with open(fname_tst, 'r') as f_in:
        for line in f_in.readlines():
            line=line.strip()
            count+=1
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
                # if cfg.TEST.AGNOSTIC:
                #     cls_boxes = boxes[inds, 4:8]
                # else:
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis],)) \
                    .astype(np.float32, copy=False)
                #keep = soft_nms(cls_dets, method=cfg.TEST.SOFT_NMS)
                cls_dets = np.hstack((cls_dets, j * np.ones((cls_dets.shape[0], 1))))
                if len(cls_dets) > 0:
                    dets_all = np.vstack((dets_all, cls_dets.copy()))
                    scores_all = np.vstack((scores_all, scores[inds].copy()))


                if is_eval:
                    keep = nms(cls_dets[:,:5].astype(np.float32), cfg.TEST.NMS)
                    cls_dets = cls_dets[keep, :]
                    all_boxes[j][count-1] = cls_dets
                    # Limit to max_per_image detections *over all classes*
                    # if max_per_image > 0:
                    #     image_scores = np.hstack([all_boxes[j][i][:, -1]
                    #                               for j in xrange(1, imdb.num_classes)])
                    #     if len(image_scores) > max_per_image:
                    #         image_thresh = np.sort(image_scores)[-max_per_image]
                    #         for j in xrange(1, imdb.num_classes):
                    #             keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    #             all_boxes[j][i] = all_boxes[j][i][keep, :]

                #all_boxes[j][i] = cls_dets
            keep = nms(dets_all[:,:5].astype(np.float32), cfg.TEST.NMS)
            # dets_all=dets_all[keep]
            data[line]['bbox']=dets_all[keep].copy()
            data[line]['cls'] = scores_all[keep].copy()

            _t['misc'].toc()
            print(count,line, _t['im_detect'].average_time,_t['misc'].average_time)
            if count % save_count==0:
                det_file = os.path.join(out_dir, 'det_%d.pkl' %count)
                with open(det_file, 'wb') as f:
                    cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
                data={}
        if is_eval:
            det_file = os.path.join(out_dir, '/../tmp/detections.pkl')
            with open(det_file, 'wb') as f:
                cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

            print 'Evaluating detections'

            imdb.evaluate_detections(all_boxes, out_dir+'/../tmp/')


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
