import os
import sys
import pickle as pkl

num_threads=15

# model_base='/home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/faster_voc_resnet101-base.prototxt'
# model_repeat='/home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/repeat_part.prototxt'
# model_out='/home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/res101_multi_thread.prototxt'

model_base='faster_voc_resnet101-base-thin.prototxt'
model_repeat='repeat_thin_part_v2.prototxt'
model_out='res101_multi_thread.prototxt'
rcnn_layer='rcnn_layer.pt'


txt_base=open(model_base,'r').read()
txt_repeat=open(model_repeat,'r').read()
txt_rcnn=open(rcnn_layer,'r').read()
txt_out=txt_base+'\n'
str_rcnn=''
for k in range(num_threads):
    str_rcnn+='top: "label_%d"\n' % k
txt_out+=txt_rcnn % str_rcnn +'\n'
a="{fc7},{fc7},{ab}".format(fc7=1,ab=2)
for k in range(num_threads):
    str_loc=txt_repeat.replace('XXX',str(k))
    txt_out+=str_loc
with open(model_out,'w') as fout:
    fout.write(txt_out)