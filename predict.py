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

image_path = "demo_pic.jpg"
classes = open("labels.txt")
