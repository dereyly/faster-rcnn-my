import os
import sys
import pickle as pkl

num_threads=15

# model_base='/home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/faster_voc_resnet101-base.prototxt'
# model_repeat='/home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/repeat_part.prototxt'
# model_out='/home/dereyly/progs/py-RFCN-priv/caffe-model/det/faster_rcnn/models/pascal_voc/resnet101-v2/res101_multi_thread.prototxt'

model_base='models/deploy_resnet101-base-nonlin.prototxt'
model_repeat='models/repeat_thin_part_nonlin.prototxt'
rcnn_layer='models/rcnn_layer_nonlin.pt'

model_out='models/deploy_res101_multi_thread_v4.prototxt'


txt_base=open(model_base,'r').read()
txt_repeat=open(model_repeat,'r').read()
txt_rcnn=open(rcnn_layer,'r').read()
str_out=''

for k in range(num_threads):
    str_loc=txt_repeat.replace('XXX',str(k))
    str_out+=str_loc
str_out+='\n'
str_rcnn=''
for k in range(num_threads):
    str_rcnn+='bottom: "cls_score_%d"\n' % k
# for k in range(num_threads):
#     str_rcnn+='bottom: "bbox_pred_%d"\n' % k
str_out+=txt_rcnn % str_rcnn +'\n'

txt_out=txt_base%str_out
with open(model_out,'w') as fout:
    fout.write(txt_out)