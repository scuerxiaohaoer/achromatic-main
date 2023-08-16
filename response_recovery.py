# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:21:48 2020

@author: Administrator

"""

import sys
#import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
from Coherent_main_function import MainFunction
from parameters import *
from datasets_processing import TrainingData
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings('ignore')
tf.reset_default_graph()

def train(): #'''训练主体函数'''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #'''硬件配置命令'''
    with tf.Session(config=config) as sess: 
        train_function = MainFunction(sess)
        sess.run(tf.global_variables_initializer())
        ckpt_path = './MHINet_ckpt/1084.ckpt'
        train_function.MHINet_restore(ckpt_path)
        train_function.train(EPOCH_NUMS)

def test_external(): #'''测试主体函数'''
    #config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        test_externel_function = MainFunction(sess)
        ckpt_path = 'all_results/ckpt_result/1200.ckpt'
        test_externel_function.restore(ckpt_path)
        test_externel_function.test_externel()

def interupt():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        interupt_function = MainFunction(sess)
        ckpt_path = 'all_results/ckpt_result/1200.ckpt'
        interupt_function.restore(ckpt_path)
        interupt_function.train(EPOCH_NUMS)
        


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please tell me more about the function you want, such as test or train')
        exit(-1)
    function = sys.argv[1]
    globals()[function](*sys.argv[2:])
    

