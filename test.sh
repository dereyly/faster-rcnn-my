#python tools/test_net.py --gpu 1 --def models/pvanet/example_train/test_2.prototxt --net output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_v2_iter_160000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#python tools/test_net.py --gpu 1 --def models/pvanet/example_train/test_5.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_5c_iter_130000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml # 0.7378

#python tools/test_net.py --gpu 1 --def models/pvanet/example_train/deploy_faster_voc_resnet101-multi-anchors.prototxt --net output/faster_rcnn_pvanet/voc_2007_trainval/res101-multi_iter_80000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml

#python tools/test_simple_v2.py --gpu 1  --def models/pvanet/example_train/test_5.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/pvanet_5c_iter_130000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml
#python tools/test_net.py  --gpu 1 --def /home/dereyly/progs/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-my.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/res101-my_iter_100000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml

#python tools/test_net.py  --gpu 0 --def=/home/dereyly/progs/pva-faster-rcnn/models/pvanet/example_train/deploy_faster_voc_resnet101-multi-anchors.prototxt --net=/home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/res101-multi_iter_100000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml

#python tools/test_net.py --gpu 1  --def models/pvanet/example_train/test_5.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval+voc_2012_trainval/pvanet_5c_iter_180000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml #0.7678

#python tools/test_net.py --gpu 0 --def models/pvanet/example_train/test_2.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval+voc_2012_trainval/pvanet_2_soft_a_iter_140000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml #0.776

#python tools/test_net.py  --gpu 1 --def /home/dereyly/progs/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-my.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval+voc_2012_trainval/res101_iter_90000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml #Mean AP = 0.8117

#python tools/test_net.py  --gpu 0 --def=/home/dereyly/progs/pva-faster-rcnn/models/pvanet/example_train/deploy_faster_voc_resnet101-multi-anchors.prototxt --net=/home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/res101-multi-2_iter_90000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml #Mean AP = 0.7652
#python tools/test_net.py  --gpu 1 --def /home/dereyly/progs/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-food.prototxt --net /home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_5180_trainval/res101_food_iter_60000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml --imdb=voc_5180_test
#python tools/test_net.py  --gpu 0 --def=/home/dereyly/progs/group_caffe/faster-rcnn-my/lib/generator/deploy_res101_multi_thread.prototxt --net=/home/dereyly/progs/py-RFCN-priv/output/faster_rcnn_end2end/voc_2007_trainval/faster_voc_resnet101-multi-anchrs_ss_iter_180000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml #Mean AP = 0.7652
#python tools/test_net.py  --gpu 0 --def=/home/dereyly/progs/group_caffe/faster-rcnn-my/lib/generator/models/deploy_res101_multi_thread.prototxt --net=/home/dereyly/progs/py-RFCN-priv/output/faster_rcnn_end2end/voc_2007_trainval/res101-multi-anchrs-v3_ss_iter_180000_sum11.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml
#python tools/test_net.py  --gpu 0 --def /home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/deploy_faster_voc_resnet101-fc2.prototxt --net /home/dereyly/progs/py-RFCN-priv/output/faster_rcnn_end2end/voc_2007_trainval/res101-multi-anchrs-v3_ss_iter_180000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml
#python tools/test_net.py  --gpu 0 --def=/home/dereyly/progs/py-RFCN-priv/generate/deploy_faster_voc_resnet101-experimental.prototxt --net=/home/dereyly/progs/py-RFCN-priv/output/faster_rcnn_end2end/voc_2007_trainval/res101-ex_ss_iter_180000.caffemodel  --cfg models/pvanet/cfgs/submit_1019.yml #0.729
python tools/test_net.py  --gpu 1 --def=/home/dereyly/progs/pva-faster-rcnn/generator/models/deploy_res101_multi_thread.prototxt --net=/home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/res101_multi_nonlin2_iter_50000.caffemodel --cfg models/pvanet/cfgs/submit_1019.yml




