root=/home/dereyly/progs/pva-faster-rcnn/
#python2 tools/train_net.py --gpu 0 --solver $root/models/pvanet/example_train/solver.prototxt --weights $root/models/pvanet/pretrained/pva9.1_pretrained_no_fc6.caffemodel --iters 180000 --cfg $root/models/pvanet/cfgs/train.yml --imdb voc_2007_trainval+voc_2012_trainval
# python2 tools/train_net.py --gpu 1    --solver=$root/models/pvanet/example_train/solver_2.prototxt --weights=/home/dereyly/progs/pva-faster-rcnn/models/priv/resnet101-v2-merge.caffemodel  --iters 120000  --cfg $root/models/pvanet/cfgs/train_101.yml --imdb voc_2007_trainval+voc_2012_trainval
#python2 tools/train_net.py --gpu 0    --solver=$root/models/pvanet/example_train/solver_2.prototxt --weights=/home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_2007_trainval/res101-multi_iter_10000.caffemodel  --iters 120000  --cfg $root/models/pvanet/cfgs/train_101.yml --imdb voc_2007_trainval
#python2 tools/train_net.py --gpu 0    --solver=$root/models/pvanet/example_train/solver_2.prototxt --weights=/home/dereyly/progs/pva-faster-rcnn/models/priv/resnet101-v2-merge.caffemodel  --iters 120000  --cfg $root/models/pvanet/cfgs/train_101_food.yml --imdb voc_5180_trainval

python2 tools/train_net.py --gpu 0    --solver=$root/models/pvanet/example_train/solver_2.prototxt --weights=/home/dereyly/progs/pva-faster-rcnn/output/faster_rcnn_pvanet/voc_5180_trainval/res101_food_iter_60000.caffemodel  --iters 120000  --cfg $root/models/pvanet/cfgs/train_101_food.yml --imdb voc_5180_trainval