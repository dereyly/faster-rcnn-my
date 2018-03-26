import os
import sys
import pickle as pkl

num_threads=3

model_base='faster_voc_resnet101-base.prototxt'
model_repeat='repeat_part.prototxt'
model_out='res101_multi_thread.prototxt'



txt_base=open(model_base,'r').read()
txt_repeat=open(model_repeat,'r').read()
txt_out=txt_base
a="{fc7},{fc7},{ab}".format(fc7=1,ab=2)
for k in range(num_threads):
    str_loc=txt_repeat.format(fc7='fc7_%d'%k, cls_score='cls_score_%d'%k, bbox_pred='bbox_pred_%d'%k)
    str_loc=str_loc.replace('[', '{').replace(']', '}')

    txt_out+=str_loc
with open(model_out,'w') as fout:
    fout.write(txt_out)