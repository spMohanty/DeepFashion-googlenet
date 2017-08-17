#!/usr/bin/env python

import os
import numpy as np

import sys
"""
If you do not have caffe root setup
caffe_root = '~/caffe/' #Path to you caffe root
sys.path.insert(0, caffe_root + 'python')
"""

import caffe
caffe.set_mode_gpu()

image_path = "demo_pic.jpg"
classes = open("labels.txt").readlines()
classes = [x.strip() for x in classes]


mean_filename=os.path.join('mean.binaryproto')
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0].mean(1).mean(1)

net_pretrained = os.path.join("snapshot_iter_8135.caffemodel")
net_model_file = os.path.join("deploy.prototxt")
Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

input_image = caffe.io.load_image(image_path)

prediction = Net.predict([input_image],oversample=False)
print "="*100
for _idx, val in enumerate(prediction[0]):
    print classes[_idx], ": ", val*100,"%"
print "="*100
print 'predicted category is {0}'.format(classes[prediction.argmax()])
