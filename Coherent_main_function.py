 #-*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:45:39 2020

@author: Administrator
"""
from parameters import *
import tensorflow as tf
from data_generator import load_data
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from data_generator import One_batch_items
from sub_files import Tee
import matplotlib.pyplot as plt
#import cv2
import random
import math
import warnings
import os
import time
import layers.optics as optics
import layers.deconv as deconv
import poppy

if not os.path.exists('all_results/mat'):
    os.makedirs('all_results/mat')

warnings.filterwarnings('ignore')

slim = tf.contrib.slim
reg = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)


def weight_init(name, shape):
    weight = tf.get_variable(name=name + '_filter_', shape=shape,
                             initializer=tf.variance_scaling_initializer(), regularizer=reg)
    #tf.add_to_collection('D_Variables',weight)
    return weight


def bias_init(name, shape):
    bias = tf.get_variable(name=name + '_bias_', shape=shape, initializer=tf.constant_initializer(0.0))
    #tf.add_to_collection('D_Variables',bias)
    return bias

def weight_init_G(name, shape):
    weight = tf.get_variable(name=name + '_filter_', shape=shape,
                             initializer=tf.variance_scaling_initializer(), regularizer=reg)
    if (weight not in tf.get_collection('G_Variables')):
        tf.add_to_collection('G_Variables',weight)
    return weight


def bias_init_G(name, shape):
    bias = tf.get_variable(name=name + '_bias_', shape=shape, initializer=tf.constant_initializer(0.0))
    if (bias not in tf.get_collection('G_Variables')):
        tf.add_to_collection('G_Variables',bias)
    return bias

def CVNet_weight_init(name,shape):
    weight = tf.get_variable(name=name + '_filter_', 
                             shape=shape,
                             initializer=tf.variance_scaling_initializer(), 
                             regularizer=reg,
                             trainable=False)
    tf.add_to_collection('CVNet_Variables',weight)
    return weight

def CVNet_bias_init(name, shape):
    bias = tf.get_variable(name=name + '_bias_', 
                           shape=shape, 
                           initializer=tf.constant_initializer(0.0),
                           trainable=False)
    tf.add_to_collection('CVNet_Variables',bias)
    return bias

def data_cross_split(datas, slice_nums, shuffle_enable):#数据切块儿'''
    datas_cross = [[][:] for index in range(slice_nums)] #生成一个空集合和对应的索引index'''
    if shuffle_enable:
        random.shuffle(datas) #随机打乱数据'''
    number = math.ceil(len(datas)/slice_nums)
    for i in range(slice_nums):
        print('\n we are spliting ', i, ' datas')
        start = i * number
        end = min(start + number, len(datas)) 
        for j in range(start, end):           #区间长度为切片数量'''
            datas_cross[i].append(datas[j])
            #print('the ', j , ' filename is', datas[j].filename)
    return datas_cross

'''percptual loss'''

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.max_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

vgg_path=scipy.io.loadmat('vgg19')
print("[i] Loaded pre-trained vgg19 parameters")
#build VGG19 to load pre-trained parameters

def build_vgg19(input):
    net={}
    vgg_layers=vgg_path['layers'][0]
    net['input']=input
    net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
    net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
    net['pool1']=build_net('pool',net['conv1_2'])
    net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
    net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
    net['pool2']=build_net('pool',net['conv2_2'])
    net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
    net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
    net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
    net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
    net['pool3']=build_net('pool',net['conv3_4'])
    net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
    net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
    net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
    net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
    net['pool4']=build_net('pool',net['conv4_4'])
    net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
    net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
    return net

def compute_l2_loss(input, output):
    return tf.losses.mean_squared_error(input, output)

def compute_percep_loss(input, output):
    input = input[:, :, :, ::-1]*255.0
    output = output[:, :, :, ::-1]*255.0
    sub_mean = tf.constant([104.0, 117.0, 123.0],dtype=tf.float32,shape=[1, 1, 1, 3],name='img_sub_mean')
    input = input - sub_mean
    output = output - sub_mean
    vgg_real=build_vgg19(output)
    vgg_fake=build_vgg19(input)
    p0=compute_l2_loss(vgg_real['input'],vgg_fake['input'])/21438
    p1=compute_l2_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/29928
    p2=compute_l2_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/49163
    p3=compute_l2_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/52520
    p4=compute_l2_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/34523/10.0
    p5=compute_l2_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])/21552*10.0
    return p0+p1+p2+p3+p4

class MainFunction:
    '''SSIM=percep, camera_response=hdm_GT_phase'''

    def __init__(self, sess=None):
        self.sess = sess
        self.dropout = tf.placeholder(tf.float32, shape=(), name='dropout')
        self.global_step = tf.Variable(0)
        #self.learning_rate = tf.train.exponential_decay(LEARNING_RATE,self.global_step,5000,0.92,staircase=False)
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.ground_truth = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name='ground_truth')
        self.reconstructed_images = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3),name='reconstructed_images')
        self.color_checker_image = tf.placeholder(tf.float32, shape=(None, 512, 512, 3),name='color_checker_image')
        self.camera_response = tf.placeholder(tf.float32, shape=(None, PHASE_SIZE, PHASE_SIZE),name='color_checker_image')
        self.camera_response = tf.where(self.camera_response >= 0, self.camera_response % (np.pi), self.camera_response)
        self.camera_response = tf.where(self.camera_response < 0, self.camera_response % (-1*np.pi), self.camera_response)
        # if self.camera_response >=0:
        #     self.camera_response = self.camera_response % (np.pi)
        # else:
        #     self.camera_response = self.camera_response % (-1*np.pi)    
        images = self.reconstructed_images
        self.is_training = tf.placeholder(tf.bool)
        
        '''Gamma'''
        if GAMMA_ENABLE:
            images = self.reconstructed_images * 1.0 / 255
            images = (tf.pow(images, 0.4545) * 255.0)
        
        '''CVNet'''
        confident_rgb = self.CVNet(self.color_checker_image, self.dropout)  
        
        #CVNet里加tf.saver
        confident_rgb_w, confident_rgb_h = map(int, confident_rgb.get_shape()[1:3])
        confidence = confident_rgb[:, :, :, 0] 
        confidence = tf.reshape(confidence, shape=[-1, confident_rgb_w * confident_rgb_h])   
        confidence = tf.nn.softmax(confidence)
        confidence = tf.reshape(confidence, shape=[-1, confident_rgb_w, confident_rgb_h, 1])      
        print('confidence shape is : ', confidence.get_shape())
        
 
        constructed_r = tf.matmul(tf.reduce_sum(confidence * confident_rgb[:,:,:,1 : BASIC_NUMS + 1], axis=(1, 2)), BASIC_FUNCTIONS.astype(np.float32), transpose_b = True)
        constructed_g = tf.matmul(tf.reduce_sum(confidence * confident_rgb[:,:,:,BASIC_NUMS + 1 : 2 * BASIC_NUMS + 1], axis=(1, 2)), BASIC_FUNCTIONS.astype(np.float32), transpose_b = True)
        constructed_b = tf.matmul(tf.reduce_sum(confidence * confident_rgb[:,:,:,2 * BASIC_NUMS + 1 : 3 * BASIC_NUMS + 1], axis=(1, 2)), BASIC_FUNCTIONS.astype(np.float32), transpose_b = True)
        print('constructed_r shape is : ', constructed_r.get_shape())
        self.constructed_response = tf.stack([constructed_r, constructed_g, constructed_b],axis = -1)      
        constructed_response_R = self.constructed_response[:,:,0]
        constructed_response_G = self.constructed_response[:,:,1]
        constructed_response_B = self.constructed_response[:,:,2] 
        
        R_list = []
        G_list = []
        B_list = []
        for wav_num in range(0,310,10):
            R_list.append(constructed_response_R[:, wav_num])
            G_list.append(constructed_response_G[:, wav_num])
            B_list.append(constructed_response_B[:, wav_num])
        
        constructed_response_R = tf.stack(R_list, axis=-1)
        constructed_response_G = tf.stack(G_list, axis=-1)
        constructed_response_B = tf.stack(B_list, axis=-1)
        
        constructed_response_R = tf.reshape(constructed_response_R,shape = [-1,1,1,31])
        constructed_response_G = tf.reshape(constructed_response_G,shape = [-1,1,1,31])
        constructed_response_B = tf.reshape(constructed_response_B,shape = [-1,1,1,31])
        
        print('The shape of down_sample response_R is: ', constructed_response_R.get_shape())
        
        constructed_response_R = tf.nn.conv2d(constructed_response_R, weight_init_G('conv_R',[1,1,31,31]), strides=[1, 1, 1, 1], padding='SAME')
        constructed_response_R = tf.nn.bias_add(constructed_response_R, bias_init_G('bias_R', [31]))
        constructed_response_G = tf.nn.conv2d(constructed_response_G, weight_init_G('conv_G',[1,1,31,31]), strides=[1, 1, 1, 1], padding='SAME')
        constructed_response_G = tf.nn.bias_add(constructed_response_G, bias_init_G('bias_G', [31]))
        constructed_response_B = tf.nn.conv2d(constructed_response_B, weight_init_G('conv_B',[1,1,31,31]), strides=[1, 1, 1, 1], padding='SAME')
        constructed_response_B = tf.nn.bias_add(constructed_response_B, bias_init_G('bias_B', [31]))
        
        '''UNet'''
        #with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn = tf.nn.relu, weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
        with tf.variable_scope("UNet"):    
            self.UNet_output_R = self.UNet(tf.expand_dims(images[:,:,:,0],axis=-1), self.dropout, 'UNet_1',self.is_training) 
            self.UNet_output_G = self.UNet(tf.expand_dims(images[:,:,:,1],axis=-1), self.dropout, 'UNet_2',self.is_training)
            self.UNet_output_B = self.UNet(tf.expand_dims(images[:,:,:,2],axis=-1), self.dropout, 'UNet_3',self.is_training)        

            
        '''Attation'''      
        self.channel_attation_R = self.channel_attation(self.UNet_output_R, 'R_channel_attation')
        self.channel_attation_G = self.channel_attation(self.UNet_output_G, 'G_channel_attation')
        self.channel_attation_B = self.channel_attation(self.UNet_output_B, 'B_channel_attation')
        self.spectral_attation_R = constructed_response_R * self.channel_attation_R
        self.spectral_attation_G = constructed_response_G * self.channel_attation_G
        self.spectral_attation_B = constructed_response_B * self.channel_attation_B  
        R_weights, G_weights, B_weights = self.compute_RGB_weights(images)
        R_weights = tf.reshape(R_weights,shape=[-1,IMAGE_SIZE,IMAGE_SIZE,1])
        G_weights = tf.reshape(G_weights,shape=[-1,IMAGE_SIZE,IMAGE_SIZE,1])
        B_weights = tf.reshape(B_weights,shape=[-1,IMAGE_SIZE,IMAGE_SIZE,1])
        self.fake_image = (R_weights * self.spectral_attation_R) + (G_weights * self.spectral_attation_G) + (B_weights * self.spectral_attation_B)
        
        self.fake_image_list = []
        for ii in range(0,spectral_num):
            self.fake_image_list.append(self.fake_image[:,:,:,ii*((31//spectral_num))])
        self.spectral_image = tf.stack(self.fake_image_list, axis=-1)
            
        RR_list = []
        GG_list = []
        BB_list = []
        for wav_num in range(0,spectral_num):
            RR_list.append(constructed_response_R[:,:,:,wav_num*(31//spectral_num)])
            GG_list.append(constructed_response_G[:,:,:,wav_num*(31//spectral_num)])
            BB_list.append(constructed_response_B[:,:,:,wav_num*(31//spectral_num)])
        if GENERATOR_ENABLE and OPTIC_ENABLE and BUILD_ENABLE:
            with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
                with tf.variable_scope("Complex", reuse=False):
                    self.complex, self.phases = self.generator(self.spectral_image, self.dropout, self.is_training, spectral_num, complex_num)
                    self.wave = self.complex[0]
                    self.phase = self.phases[0]
                    for i in range(1, int(spectral_num)):
                        self.wave = tf.concat([self.wave, self.complex[i]], axis=3)
                        self.phase = tf.concat([self.phase, self.phases[i]], axis=3)
                    #self.optics_image = tf.zeros_like(tf.cast(self.complex[0], tf.float32))    
                with tf.variable_scope("optical_model", reuse=False):
                    
                    self.optics_sensor_image, self.psf, self.lens, self.target_depth, self.intensity_psf, self.focal = self.optics_model(self.wave, '_lens_1')
                    recover_amp_list = []
                    for q in range(0,spectral_num):
                        channel_q = self.optics_sensor_image[:,:,:,(q*complex_num):((q+1)*complex_num)]
                        channel_sum = tf.reduce_sum(channel_q, axis=-1)
                        recover_amp_list.append(channel_sum)
                    self.optics_sensor_image = tf.stack(recover_amp_list, axis=-1)
                    response_R = tf.stack(RR_list, axis=-1)
                    response_G = tf.stack(GG_list, axis=-1)
                    response_B = tf.stack(BB_list, axis=-1)
                    response_R = tf.reshape(response_R,shape = [-1,1,1,spectral_num])
                    response_G = tf.reshape(response_G,shape = [-1,1,1,spectral_num])
                    response_B = tf.reshape(response_B,shape = [-1,1,1,spectral_num])
                    self.achromatic_R = self.optics_sensor_image * response_R
                    self.achromatic_R = tf.reduce_sum(self.achromatic_R, axis=3)
                    self.achromatic_G = self.optics_sensor_image * response_G
                    self.achromatic_G = tf.reduce_sum(self.achromatic_G, axis=3)
                    self.achromatic_B = self.optics_sensor_image * response_B
                    self.achromatic_B = tf.reduce_sum(self.achromatic_B, axis=3)
                    self.achromatic_image = tf.stack([self.achromatic_R,self.achromatic_G,self.achromatic_B], axis=3)
                    self.recovered_pahse = self.camera_response
                
        '''Loss'''
        self.phase_MSE_loss = compute_l2_loss(self.camera_response, self.camera_response)
        self.MSE_loss = compute_l2_loss(self.achromatic_image, self.ground_truth)
        self.percep_loss = compute_percep_loss(self.achromatic_image, self.ground_truth) * 2
        self.loss = self.MSE_loss + self.percep_loss + self.phase_MSE_loss

        scalar_summaries_G = []
        scalar_summaries_G.append(tf.summary.scalar('G_loss', self.loss))
        self.scalar_summaries_G = tf.summary.merge(scalar_summaries_G)
        loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.total_loss = self.loss + (loss_reg * 0)
        # self.D_Variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator/batch_normalization') + tf.get_collection('D_Variables')
        self.G_Variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='UNet/bn') + tf.get_collection('G_Variables')
        self.MHINet_Variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='UNet/bn') + tf.get_collection('G_Variables') + tf.get_collection('CVNet_Variables')
        print(self.MHINet_Variables)
        '''Optimizer'''
        self.train_step_adam_G = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                                     global_step = self.global_step)
        #var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='optical_model')+tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='UNet')
        tf.global_variables_initializer()
        self.CVNet_saver = tf.train.Saver(tf.get_collection('CVNet_Variables'),max_to_keep=MAX_TO_KEEP)
        self.MHINet_saver = tf.train.Saver(self.MHINet_Variables,max_to_keep=MAX_TO_KEEP)
        self.saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
        
        '''预处理'''
        self.raw_images_fake = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name='raw_images_fake')
        self.raw_images_real = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name='raw_images_real')
        edit_images_fake = self.raw_images_fake
        edit_images_real = self.raw_images_real

        if PREPROCESS_ENABLE:
            if RANDOM_ROTATE_ENABLE:
                random_angles = tf.random.uniform(shape=(tf.shape(edit_images_fake)[0],),
                                                  minval=-ROTATE_MAX_ANGLE * np.pi / 180,
                                                  maxval=ROTATE_MAX_ANGLE * np.pi / 180)

                edit_images_fake = tf.contrib.image.transform(edit_images_fake,tf.contrib.image.angles_to_projective_transforms(random_angles, tf.cast(tf.shape(edit_images_fake)[1], tf.float32), tf.cast(tf.shape(edit_images_fake)[2], tf.float32)))
                edit_images_real = tf.contrib.image.transform(edit_images_real,tf.contrib.image.angles_to_projective_transforms(random_angles, tf.cast(tf.shape(edit_images_real)[1], tf.float32), tf.cast(tf.shape(edit_images_real)[2], tf.float32)))
            if RANDOM_CROP_ENABLE:
                boxes = []
                for box_count in range(BATCH_SIZE):
                    xmax = 1 - random.random() * (1 - CROP_LEAST_COVER_PERCENTAGE)
                    ymax = 1 - random.random() * (1 - CROP_LEAST_COVER_PERCENTAGE)
                    start = max(0, xmax - ymax * IMAGE_SIZE / IMAGE_SIZE)
                    end = ymax - CROP_LEAST_COVER_PERCENTAGE
                    xself.psfmin = start + random.random() * (end - start)
                    ymin = ymax - (xmax - xmin) * IMAGE_SIZE / IMAGE_SIZE
                    boxes.append([ymin, xmin, ymax, xmax])
                edit_images_fake = tf.image.crop_and_resize(edit_images_fake, boxes, box_ind=list(range(0, BATCH_SIZE)),
                                                       crop_size=(IMAGE_SIZE, IMAGE_SIZE))
                edit_images_real = tf.image.crop_and_resize(edit_images_real, boxes, box_ind=list(range(0, BATCH_SIZE)),
                                                       crop_size=(IMAGE_SIZE, IMAGE_SIZE))                                                       
        self.edit_images_fake = edit_images_fake
        self.edit_images_real = edit_images_real
        all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optics_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='optical_model')
        UNet_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='UNet')
        param = sum([tf.keras.backend.count_params(var) for var in all_var])
        param1 = sum([tf.keras.backend.count_params(var) for var in optics_var])
        param2 = sum([tf.keras.backend.count_params(var) for var in UNet_var])
        print('param_num is ************************', param, param1, param2) 
        
    def generator(self, images, dropout, is_training, spectral_num, complex_num):
        # images = input_crop(images, 1356, 1, 1)
        #complex_wave = tf.cast(tf.sqrt(images), tf.complex64)
        conv1 = slim.conv2d(images, 32, [3, 3], activation_fn = None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)
        conv1 = slim.conv2d(conv1, 32, [3, 3], activation_fn = None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], activation_fn = None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.relu(conv2)
        conv2 = slim.conv2d(conv2, 64, [3, 3], activation_fn = None)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.relu(conv2)
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], activation_fn = None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)
        conv3 = slim.conv2d(conv3, 128, [3, 3], activation_fn = None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], activation_fn = None)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        conv4 = tf.nn.relu(conv4)
        conv4 = slim.conv2d(conv4, 256, [3, 3], activation_fn = None)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        conv4 = tf.nn.relu(conv4)
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], activation_fn = None)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        conv5 = tf.nn.relu(conv5)
        conv5 = slim.conv2d(conv5, 512, [3, 3], activation_fn = None)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        conv5 = tf.nn.relu(conv5)

        #conv5 = self.channel_attation_x(conv5, 'generator_channel')

        up6 = self.upsample_and_concat('up6',conv5, conv4, 256, 512, add_collection=False)
        conv6 = tf.layers.batch_normalization(up6, training=is_training)
        conv6 = slim.conv2d(conv6, 256, [3, 3], activation_fn = None)
        conv6 = tf.nn.relu(conv6)
        conv6 = slim.conv2d(conv6, 256, [3, 3], activation_fn = None)
        conv6 = tf.nn.relu(conv6)

        up7 = self.upsample_and_concat('up7',conv6, conv3, 128, 256, add_collection=False)
        conv7 = tf.layers.batch_normalization(up7, training=is_training)
        conv7 = slim.conv2d(conv7, 128, [3, 3], activation_fn = None)
        conv7 = tf.nn.relu(conv7)
        conv7 = slim.conv2d(conv7, 128, [3, 3], activation_fn = None)
        conv7 = tf.nn.relu(conv7)

        up8 = self.upsample_and_concat('up8',conv7, conv2, 64, 128, add_collection=False)
        conv8 = tf.layers.batch_normalization(up8, training=is_training)
        conv8 = slim.conv2d(conv8, 64, [3, 3], activation_fn = None)
        conv8 = tf.nn.relu(conv8)
        conv8 = slim.conv2d(conv8, 64, [3, 3], activation_fn = None)
        conv8 = tf.nn.relu(conv8)

        up9 = self.upsample_and_concat('up9',conv8, conv1, 32, 64, add_collection=False)
        conv9 = tf.layers.batch_normalization(up9, training=is_training)
        conv9 = slim.conv2d(conv9, 32, [3, 3], activation_fn = None)
        conv9 = tf.nn.relu(conv9)
        conv9 = slim.conv2d(conv9, 32, [3, 3], activation_fn = None)
        conv9 = tf.nn.relu(conv9)
        print("conv9.shape:{}".format(conv9.get_shape()))
        conv9 = tf.nn.dropout(conv9, dropout)        
        conv10 = slim.conv2d(conv9, spectral_num*complex_num*2, [1, 1], activation_fn = None)
        complex_wave, phases = self.coherent_propogate(conv10, spectral_num, complex_num)
        print('*********************************************')
        # phase_and_amp = conv9
        # amp = phase_and_amp[:,:,:,0:(spectral_num*complex_num)]
        # phase_real_part = phase_and_amp[:,:,:,(spectral_num*complex_num):(spectral_num*complex_num*2)]
        
        # phase = optics.compl_exp_tf(phase_real_part, dtype=tf.complex64) 
        # #phase_real_part = phase_real_part % (2 * np.pi)
        # amp = tf.cast(tf.abs(amp), tf.complex64)
        # complex_wave = tf.multiply(amp, phase)
        # #conv10 = slim.conv2d(conv9, phase_num * 2, [1, 1], activation_fn = None) # phase
        # #conv10 = np.pi*tf.nn.tanh(conv10)
        # #phase = np.pi * tf.nn.tanh(phase)
        # # phase = input_crop(conv10, 1356, 1, 1)  
        # # images = input_crop(images, 1356, 1, 1)
        # #complex_waves = self.coherent_propogate(conv10)
        # '''complex wavefront'''
        # # images = tf.cast(tf.sqrt(images), tf.complex64)
        # # phase = optics.compl_exp_tf(conv10,dtype=tf.complex64)
        # # complex_wave = tf.multiply(images,phase)
        return complex_wave, phases
    
    def optics_model(self, image, lens_index):
        '''image: complex wavefront'''
        # creat zernike polynomial 231 terms
        if not os.path.exists('zernike_volume_%d.npy'%wave_resolution[0]):
            zernike_volume = optics.get_zernike_volume(resolution=wave_resolution[0], n_terms=131).astype(np.float32)
            np.save('zernike_volume_%d.npy'%wave_resolution[0], zernike_volume)
        else:
            zernike_volume = np.load('zernike_volume_%d.npy' % wave_resolution[0])
        input_img = image

        
        target_depth_initializer = tf.constant_initializer(obj_distance)
        target_depth = tf.get_variable(name="target_depth"+lens_index,
                                            shape=(),
                                            dtype=tf.float32,
                                            trainable=False,
                                            initializer=target_depth_initializer)
        target_depth = tf.abs(target_depth) # Enforce that depth is positive.
        tf.add_to_collection('Optical_Variables',target_depth)
        # sensor_distance = 63.6e-3
        # tf.summary.scalar('target_depth', target_depth)
        # sensor_distance_initializer = tf.constant_initializer(63.6e-3)
        # sensor_distance = tf.get_variable(name="sensor_diatance",
        #                                     shape=(),
        #                                     dtype=tf.float64,
        #                                     trainable=True,
        #                                     initializer=sensor_distance_initializer)
        # tf.add_to_collection('Optical_Variables',sensor_distance)
        '''
        sensor_distance_initializer = tf.constant_initializer(35.5e-3)
        sensor_distance = tf.get_variable(name="sensor_distance",
                                           shape=(),
                                           dtype=tf.float64,
                                           trainable=True,
                                           initializer=sensor_distance_initializer)
        '''
        # sensor_distance = 35.5e-3
        # sensor_distance = tf.square(sensor_distance) # Enforce that depth is positive.
        # tf.summary.scalar('sensor_distance', sensor_distance)
       
        '''simulate different depths' psf '''
        all_depths = tf.convert_to_tensor([1 / 2, 1 / 1.5, 1 / 1, 1 / 0.5, 1000], tf.float32)

        depth_bins = []
        for i in range(3):
            depth_idx = tf.multinomial(tf.log([5 * [1 / 5]]), num_samples=1)
            depth_bins.append(all_depths[tf.cast(depth_idx[0][0], tf.int32)])
            
        test_depth = np.concatenate(
            [np.ones((patch_size // len(depth_bins), patch_size)) * i for i in range(len(depth_bins))], axis=0)[:, :, None]

        depth_map = np.expand_dims(test_depth, axis = 0)
        
        optical_system = optics.ZernikeSystem(zernike_volume=zernike_volume,
                    target_distance=target_depth, # object distance /meter
                    wave_resolution=wave_resolution,
                    upsample=False, #image resolution does not match the wave resolution
                    wave_lengths=wave_lengths,
                    sensor_resolution=(patch_size, patch_size),# image resolution
                    height_tolerance=0, # lens fabrication error
                    refractive_idcs=refractive_idcs,
                    input_sample_interval=sampling_interval, #pixel size
                    sensor_distance=sensor_distance, #image distance
                    depth_bins=depth_bins,
                    focal_length=focal_length,
                    lens_index=lens_index) # object distance

            # We want to be robust to the noise level. Thus we pick a noise level at random.
        # noise_sigma = tf.random_uniform(minval=0.001, maxval=0.02, shape=[])
        # gaussian noise_sigma = 0
        sensor_img = optical_system.get_sensor_img(input_img=input_img,
                                                       noise_sigma=0,
                                                       depth_dependent=False,
                                                       depth_map=depth_map)
        output_image = tf.cast(sensor_img, tf.float32)
        psf_tar = tf.transpose(optical_system.psfs[0],[2,0,1,3])
        lens = optical_system.height_map
        intensity_psf = optical_system.intensity_psf
        focal = optical_system.focal
        '''
            # Now deconvolve
        pad_width = output_image.shape.as_list()[1] // 2

        output_image = tf.pad(output_image, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])
        output_image = deconv.inverse_filter(output_image, output_image, optical_system.target_psf,
                                                 init_gamma=0)
        output_image = output_image[:, pad_width:-pad_width, pad_width:-pad_width, :]
        '''
        return output_image, psf_tar, lens, target_depth, intensity_psf, focal 
    '''build the generator to reconstruct complex wavefront from the input intensity objective distribution'''
    
    
    def coherent_propogate(self, phase_and_amp, spectral_num, complex_num):
        #image = tf.pow(image, 2.2)
        #image = tf.cast(image, tf.complex64)
        complex_waves = []
        phases = []
        # weight = tf.get_variable('pahse_weight', shape=[phase_num/3,], initializer=tf.constant_initializer((1/phase_num)*0))
        # weight = tf.clip_by_value(weight, 0, 1)
        #weight = tf.cast(weight, tf.complex64)
        for i in range(0, int(spectral_num)):
            phase_and_amp_index = phase_and_amp[:,:,:,(complex_num*2)*i:(complex_num*2)*(i+1)]
            phase_real = phase_and_amp_index[:,:,:,0:complex_num]
            amp = tf.cast(tf.abs(phase_and_amp_index[:,:,:,complex_num:complex_num*2]), tf.complex64)
            phase = optics.compl_exp_tf(phase_real, dtype=tf.complex64)
            complex_wave = tf.multiply(amp, phase)
            complex_waves.append(complex_wave)
            phases.append(phase_real)
        return complex_waves, phases
    
    def build_phase(self, images, dropout, is_training):
        
       # conv1 = slim.conv2d(images, 3, [1, 1], activation_fn = None)    
        
        conv1 = slim.conv2d(images, 32, [3, 3], activation_fn = None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)
        conv1 = slim.conv2d(conv1, 32, [3, 3], activation_fn = None)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        # conv2 = slim.conv2d(pool1, 64, [3, 3], activation_fn = None)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        # conv2 = tf.nn.relu(conv2)
        # conv2 = slim.conv2d(conv2, 64, [3, 3], activation_fn = None)
        # conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        # conv2 = tf.nn.relu(conv2)
        # pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool1, 64, [3, 3], activation_fn = None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)
        conv3 = slim.conv2d(conv3, 64, [3, 3], activation_fn = None)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.relu(conv3)
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        # conv4 = slim.conv2d(pool3, 256, [3, 3], activation_fn = None)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        # conv4 = tf.nn.relu(conv4)
        # conv4 = slim.conv2d(conv4, 256, [3, 3], activation_fn = None)
        # conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        # conv4 = tf.nn.relu(conv4)
        # pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool3, 128, [3, 3], activation_fn = None)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        conv5 = tf.nn.relu(conv5)
        conv5 = slim.conv2d(conv5, 128, [3, 3], activation_fn = None)
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
        conv5 = tf.nn.relu(conv5)
        '''attention'''
        #conv5 = self.channel_attation_x(conv5, 'build_channel')
        #conv5 = self.spatial_attation(conv5, 'spatial')

        # up6 = self.upsample_and_concat('up66',conv5, conv4, 256, 512, add_collection=False)
        # conv6 = tf.layers.batch_normalization(up6, training=is_training)
        # conv6 = slim.conv2d(conv6, 256, [3, 3], activation_fn = None)
        # conv6 = tf.nn.relu(conv6)
        # conv6 = slim.conv2d(conv6, 256, [3, 3], activation_fn = None)
        # conv6 = tf.nn.relu(conv6)

        up7 = self.upsample_and_concat('up77',conv5, conv3, 64, 128, add_collection=False)
        conv7 = tf.layers.batch_normalization(up7, training=is_training)
        conv7 = slim.conv2d(conv7, 64, [3, 3], activation_fn = None)
        conv7 = tf.nn.relu(conv7)
        conv7 = slim.conv2d(conv7, 64, [3, 3], activation_fn = None)
        conv7 = tf.nn.relu(conv7)

        # up8 = self.upsample_and_concat('up88',conv7, conv2, 64, 128, add_collection=False)
        # conv8 = tf.layers.batch_normalization(up8, training=is_training)
        # conv8 = slim.conv2d(conv8, 64, [3, 3], activation_fn = None)
        # conv8 = tf.nn.relu(conv8)
        # conv8 = slim.conv2d(conv8, 64, [3, 3], activation_fn = None)
        # conv8 = tf.nn.relu(conv8)

        up9 = self.upsample_and_concat('up99',conv7, conv1, 32, 64, add_collection=False)
        conv9 = tf.layers.batch_normalization(up9, training=is_training)
        conv9 = slim.conv2d(conv9, 31, [3, 3], activation_fn = None)
        conv9 = tf.nn.relu(conv9)
        conv9 = slim.conv2d(conv9, 31, [3, 3], activation_fn = None)
        conv9 = tf.nn.relu(conv9)
        print("conv9.shape:{}".format(conv9.get_shape()))
        
        conv9 = tf.nn.dropout(conv9, dropout)        
        conv10 = slim.conv2d(conv9, 1, [1, 1], activation_fn = None)
        # conv10 = tf.where(conv10 >= 0, conv10 % (np.pi), conv10)
        # conv10 = tf.where(conv10 < 0, conv10 % (-1*np.pi), conv10)
        
        return conv10
    
    def compute_RGB_weights(self, RGB_image):
        R = RGB_image[:,:,:,0]
        G = RGB_image[:,:,:,1]
        B = RGB_image[:,:,:,2]
        R_weights = (R*R) + (R*G) + (R*B)
        G_weights = (G*R) + (G*G) + (G*B)
        B_weights = (B*R) + (B*G) + (B*B)
        
        weight_block = tf.stack([R_weights,G_weights,B_weights], axis=-1)
        weight_block = tf.nn.softmax(weight_block, axis=-1) #沿通道维对每一个像素位置做softmax
        
        R_attation_weights = weight_block[:,:,:,0]
        G_attation_weights = weight_block[:,:,:,1]
        B_attation_weights = weight_block[:,:,:,2]
        
        return R_attation_weights,G_attation_weights,B_attation_weights
          
    def compute_percep_D_loss(self, image_real, image_fake): #动态感知损失（未使用）
        real=self.Discriminator(image_real)
        fake=self.Discriminator(image_fake)
        loss_0=compute_l2_loss(image_real, image_fake)
        loss_1=compute_l2_loss(real['pool1'], fake['pool1'])
        loss_2=compute_l2_loss(real['pool2'], fake['pool2'])
        loss_3=compute_l2_loss(real['pool3'], fake['pool3'])
        loss_4=compute_l2_loss(real['pool4'], fake['pool4'])
        loss_5=compute_l2_loss(real['pool5'], fake['pool5'])
        loss_6=compute_l2_loss(real['pool6'], fake['pool6'])
        loss_7=compute_l2_loss(real['pool7'], fake['pool7'])
        loss_8=compute_l2_loss(real['pool8'], fake['pool8'])
        loss_9=compute_l2_loss(real['pool9'], fake['pool9'])
        return (loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7+loss_8+loss_9)/1000
           
    def CVNet(self, images, dropout):        
        '''卷积、池化主体函数，输入为batch_size*512*512*3，输出为batch_size*w*h*(1+3*381)'''                                                      
        net = {}
        net['input'] = images
        print('net[input] shape is : ', net['input'].get_shape())
        '''512*512*3'''
        net['conv1'] = tf.nn.conv2d(net['input'], CVNet_weight_init('conv1',[3,3,3,32]), strides=[1, 1, 1, 1], padding='VALID')
        net['conv1'] = tf.nn.bias_add(net['conv1'], CVNet_bias_init('conv1', [32]))
        net['relu1'] = tf.nn.relu(net['conv1'])
        net['pool1'] = tf.nn.max_pool(net['relu1'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('net[pool1] shape is : ', net['pool1'].get_shape())
        '''255*255*32'''
        net['conv2'] = tf.nn.conv2d(net['pool1'], CVNet_weight_init('conv2',[3,3,32,64]), strides=[1, 1, 1, 1], padding='VALID')
        net['conv2'] = tf.nn.bias_add(net['conv2'], CVNet_bias_init('conv2', [64]))
        net['relu2'] = tf.nn.relu(net['conv2'])
        net['pool2'] = tf.nn.max_pool(net['relu2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('net[pool2] shape is : ', net['pool2'].get_shape())
        '''127*127*64'''
        net['conv3'] = tf.nn.conv2d(net['pool2'], CVNet_weight_init('conv3',[3,3,64,128]), strides=[1, 1, 1, 1], padding='VALID')
        net['conv3'] = tf.nn.bias_add(net['conv3'], CVNet_bias_init('conv3', [128]))
        net['relu3'] = tf.nn.relu(net['conv3'])
        net['pool3'] = tf.nn.max_pool(net['relu3'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('net[pool3] shape is : ', net['pool3'].get_shape())
        '''63*63*128'''
        net['conv4'] = tf.nn.conv2d(net['pool3'], CVNet_weight_init('conv4',[3,3,128,256]), strides=[1, 1, 1, 1], padding='VALID')
        net['conv4'] = tf.nn.bias_add(net['conv4'], CVNet_bias_init('conv4', [256]))
        net['relu4'] = tf.nn.relu(net['conv4'])
        net['pool4'] = tf.nn.max_pool(net['relu4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('net[pool4] shape is : ', net['pool4'].get_shape())
        '''31*31*256'''
        net['conv5'] = tf.nn.conv2d(net['pool4'], CVNet_weight_init('conv5',[3,3,256,CHANNELS]), strides=[1, 1, 1, 1], padding='VALID')
        net['conv5'] = tf.nn.bias_add(net['conv5'], CVNet_bias_init('conv5', [CHANNELS]))
        net['relu5'] = tf.nn.relu(net['conv5'])
        net['pool5'] = tf.nn.max_pool(net['relu5'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('net[pool5] shape is : ', net['pool5'].get_shape())
        '''15*15*381'''
        net['dropout6'] = tf.nn.dropout(net['pool5'], dropout)
        output_channel = 3 * BASIC_NUMS + 1
        full_shape = net['dropout6'].get_shape()
        net['conv7'] = tf.nn.conv2d(net['dropout6'], CVNet_weight_init('conv7',[3, 3, full_shape[3], output_channel]), strides=[1, 1, 1, 1], padding='VALID')
        net['conv7'] = tf.nn.bias_add(net['conv7'], CVNet_bias_init('conv7', [output_channel]))
        print('net[conv7] shape is : ', net['conv7'].get_shape())
        '''13*13*(1+3*381)'''
        confident_rgb = net['conv7']
        return confident_rgb
    
    def UNet(self, images, dropout, UNet_name, is_training=True):
        net = {}
        net['input'] = images
        print('Generator[input] shape is : ', net['input'].get_shape())

        net['conv11'] = tf.nn.conv2d(net['input'], weight_init_G('conv11' + UNet_name,[3,3,1,32]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv11'] = tf.nn.bias_add(net['conv11'], bias_init_G('convG11' + UNet_name, [32]))
        net['conv11'] = tf.layers.batch_normalization(net['conv11'], name='bn1' + UNet_name, training=is_training)
        net['relu11'] = tf.nn.relu(net['conv11'])
        net['conv12'] = tf.nn.conv2d(net['relu11'], weight_init_G('conv12' + UNet_name,[3,3,32,32]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv12'] = tf.nn.bias_add(net['conv12'], bias_init_G('convG12' + UNet_name, [32]))
        net['conv12'] = tf.layers.batch_normalization(net['conv12'],name='bn2' + UNet_name, training=is_training)
        net['relu12'] = tf.nn.relu(net['conv12'])
        net['pool1'] = tf.nn.max_pool(net['relu12'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        net['conv21'] = tf.nn.conv2d(net['pool1'], weight_init_G('conv21' + UNet_name,[3,3,32,64]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv21'] = tf.nn.bias_add(net['conv21'], bias_init_G('convG21' + UNet_name, [64]))
        net['conv21'] = tf.layers.batch_normalization(net['conv21'],name='bn3' + UNet_name, training=is_training)
        net['relu21'] = tf.nn.relu(net['conv21'])
        net['conv22'] = tf.nn.conv2d(net['relu21'], weight_init_G('conv22' + UNet_name,[3,3,64,64]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv22'] = tf.nn.bias_add(net['conv22'], bias_init_G('convG22' + UNet_name, [64]))
        net['conv22'] = tf.layers.batch_normalization(net['conv22'],name='bn5' + UNet_name, training=is_training)
        net['relu22'] = tf.nn.relu(net['conv22'])
        net['pool2'] = tf.nn.max_pool(net['relu22'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        net['conv31'] = tf.nn.conv2d(net['pool2'], weight_init_G('conv31' + UNet_name,[3,3,64,128]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv31'] = tf.nn.bias_add(net['conv31'], bias_init_G('convG31' + UNet_name, [128]))
        net['conv31'] = tf.layers.batch_normalization(net['conv31'],name='bn6' + UNet_name, training=is_training)
        net['relu31'] = tf.nn.relu(net['conv31'])
        net['conv32'] = tf.nn.conv2d(net['relu31'], weight_init_G('conv32' + UNet_name,[3,3,128,128]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv32'] = tf.nn.bias_add(net['conv32'], bias_init_G('convG32' + UNet_name, [128]))
        net['conv32'] = tf.layers.batch_normalization(net['conv32'],name='bn7' + UNet_name, training=is_training)
        net['relu32'] = tf.nn.relu(net['conv32'])
        net['pool3'] = tf.nn.max_pool(net['relu32'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        net['conv41'] = tf.nn.conv2d(net['pool3'], weight_init_G('conv41' + UNet_name,[3,3,128,256]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv41'] = tf.nn.bias_add(net['conv41'], bias_init_G('convG41' + UNet_name, [256]))
        net['conv41'] = tf.layers.batch_normalization(net['conv41'],name='bn8' + UNet_name, training=is_training)
        net['relu41'] = tf.nn.relu(net['conv41'])
        net['conv42'] = tf.nn.conv2d(net['relu41'], weight_init_G('conv42' + UNet_name,[3,3,256,256]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv42'] = tf.nn.bias_add(net['conv42'], bias_init_G('convG42' + UNet_name, [256]))
        net['conv42'] = tf.layers.batch_normalization(net['conv42'],name='bn9' + UNet_name, training=is_training)
        net['relu42'] = tf.nn.relu(net['conv42'])
        net['pool4'] = tf.nn.max_pool(net['relu42'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        net['conv51'] = tf.nn.conv2d(net['pool4'], weight_init_G('conv51' + UNet_name,[3,3,256,512]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv51'] = tf.nn.bias_add(net['conv51'], bias_init_G('convG51' + UNet_name, [512]))
        net['conv51'] = tf.layers.batch_normalization(net['conv51'],name='bn10' + UNet_name, training=is_training)
        net['relu51'] = tf.nn.relu(net['conv51'])
        net['conv52'] = tf.nn.conv2d(net['relu51'], weight_init_G('conv52' + UNet_name,[3,3,512,512]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv52'] = tf.nn.bias_add(net['conv52'], bias_init_G('convG52' + UNet_name, [512]))
        net['conv52'] = tf.layers.batch_normalization(net['conv52'],name='bn11' + UNet_name, training=is_training)
        net['relu52'] = tf.nn.relu(net['conv52'])

        net['relu52'] = self.spatial_attation(net['relu52'],'inside_weight' + UNet_name)

        net['up_1'] = self.upsample_and_concat('up_1' + UNet_name,net['relu52'], net['relu42'], 256, 512)
        net['conv61'] = tf.nn.conv2d(net['up_1'], weight_init_G('conv61' + UNet_name,[3,3,512,256]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv61'] = tf.nn.bias_add(net['conv61'], bias_init_G('convG61' + UNet_name, [256]))
        net['conv61'] = tf.layers.batch_normalization(net['conv61'],name='bn12' + UNet_name, training=is_training)
        net['relu61'] = tf.nn.relu(net['conv61'])
        net['conv62'] = tf.nn.conv2d(net['relu61'], weight_init_G('conv62' + UNet_name,[3,3,256,256]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv62'] = tf.nn.bias_add(net['conv62'], bias_init_G('convG62' + UNet_name, [256]))
        net['conv62'] = tf.layers.batch_normalization(net['conv62'],name='bn13' + UNet_name, training=is_training)
        net['relu62'] = tf.nn.relu(net['conv62'])
        
        net['up_2'] = self.upsample_and_concat('up_2' + UNet_name,net['relu62'], net['relu32'], 128, 256)
        net['conv71'] = tf.nn.conv2d(net['up_2'], weight_init_G('conv71' + UNet_name,[3,3,256,128]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv71'] = tf.nn.bias_add(net['conv71'], bias_init_G('convG71' + UNet_name, [128]))
        net['conv71'] = tf.layers.batch_normalization(net['conv71'],name='bn14' + UNet_name, training=is_training)
        net['relu71'] = tf.nn.relu(net['conv71'])
        net['conv72'] = tf.nn.conv2d(net['relu71'], weight_init_G('conv72' + UNet_name,[3,3,128,128]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv72'] = tf.nn.bias_add(net['conv72'], bias_init_G('convG72' + UNet_name, [128]))
        net['conv72'] = tf.layers.batch_normalization(net['conv72'],name='bn15' + UNet_name, training=is_training)
        net['relu72'] = tf.nn.relu(net['conv72'])
        
        net['up_3'] = self.upsample_and_concat('up_3' + UNet_name,net['relu72'], net['relu22'], 64, 128)
        net['conv81'] = tf.nn.conv2d(net['up_3'], weight_init_G('conv81' + UNet_name,[3,3,128,64]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv81'] = tf.nn.bias_add(net['conv81'], bias_init_G('convG81' + UNet_name, [64]))
        net['conv81'] = tf.layers.batch_normalization(net['conv81'],name='bn16' + UNet_name, training=is_training)
        net['relu81'] = tf.nn.relu(net['conv81'])
        net['conv82'] = tf.nn.conv2d(net['relu81'], weight_init_G('conv82' + UNet_name,[3,3,64,64]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv82'] = tf.nn.bias_add(net['conv82'], bias_init_G('convG82' + UNet_name, [64]))
        net['conv82'] = tf.layers.batch_normalization(net['conv82'],name='17' + UNet_name, training=is_training)
        net['relu82'] = tf.nn.relu(net['conv82'])

        net['up_4'] = self.upsample_and_concat('up_4' + UNet_name,net['relu82'], net['relu12'], 32, 64)
        net['conv91'] = tf.nn.conv2d(net['up_4'], weight_init_G('conv91' + UNet_name,[3,3,64,31]), strides=[1, 1, 1, 1], padding='SAME')
        net['conv91'] = tf.nn.bias_add(net['conv91'], bias_init_G('convG91' + UNet_name, [31]))
        # net['conv91'] = tf.layers.batch_normalization(net['conv91'], training=is_training)
        # net['relu91'] = tf.nn.relu(net['conv91'])
        # net['conv92'] = tf.nn.conv2d(net['relu91'], weight_init_G('conv92',[3,3,32,32]), strides=[1, 1, 1, 1], padding='SAME')
        # net['conv92'] = tf.nn.bias_add(net['conv92'], bias_init_G('convG92', [32]))
        # net['conv92'] = tf.layers.batch_normalization(net['conv92'], training=is_training)
        # net['relu92'] = tf.nn.relu(net['conv92'])

        # net['relu92'] = tf.nn.dropout(net['relu92'],dropout)
        # net['conv100'] = tf.nn.conv2d(net['relu92'], weight_init_G('conv100',[1,1,32,3]), strides=[1, 1, 1, 1], padding='SAME')
        # net['conv100'] = tf.nn.bias_add(net['conv100'], bias_init_G('convG100', [3]))
 
        print('Generator[output] shape is : ', net['conv91'].get_shape())
        return net['conv91']
        

    def upsample_and_concat(self,name,x1, x2, output_channels, in_channels, add_collection=True):
        pool_size = 2
        deconv_filter = tf.Variable(
            tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02),name=name)
        if add_collection:
            tf.add_to_collection('G_Variables',deconv_filter)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
        return deconv_output
    
    def spatial_attation(self,input_block,weight_name):
        w = input_block.get_shape()[1]
        h = input_block.get_shape()[2]
        channel_num = input_block.get_shape()[3]
        
        Q = tf.nn.conv2d(input_block, weight_init_G('spatial_conv1' + weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        Q = tf.nn.bias_add(Q, bias_init_G('spatial_bias1' + weight_name, [channel_num]))
        K = tf.nn.conv2d(input_block, weight_init_G('spatial_conv2'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        K = tf.nn.bias_add(K, bias_init_G('spatial_bias2' + weight_name, [channel_num]))
        V = tf.nn.conv2d(input_block, weight_init_G('spatial_conv3'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        V = tf.nn.bias_add(V, bias_init_G('spatial_bias3' + weight_name, [channel_num]))
        
        Q = tf.reshape(Q,[-1, w * h, channel_num])
        K = tf.reshape(K,[-1, w * h, channel_num])
        K = tf.transpose(K, [0,2,1])
        V = tf.reshape(V,[-1, w * h, channel_num])
        V = tf.transpose(V, [0,2,1])
        print('The shape of K is',K.get_shape())
        
        spatial_weight = tf.nn.softmax(tf.matmul(Q,K))
        spatial_weight = tf.transpose(spatial_weight, [0,2,1])
        print('The shpae of spatial_weight is', spatial_weight.get_shape())
        
        weighted_block = tf.matmul(V,spatial_weight)
        weighted_block = tf.transpose(weighted_block)
        weighted_block = tf.reshape(weighted_block,[-1, w, h, channel_num])
        print('The shape of weighted_block is',weighted_block.get_shape())
        
        spatial_attation_block = (input_block + weighted_block)
        return spatial_attation_block
    
    def channel_attation(self,input_block,weight_name):
        w = input_block.get_shape()[1]
        h = input_block.get_shape()[2]
        channel_num = input_block.get_shape()[3]
        
        Q = tf.nn.conv2d(input_block, weight_init_G('channel_conv1' + weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        Q = tf.nn.bias_add(Q, bias_init_G('channel_bias1' + weight_name, [channel_num]))
        K = tf.nn.conv2d(input_block, weight_init_G('channel_conv2'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        K = tf.nn.bias_add(K, bias_init_G('channel_bias2' + weight_name, [channel_num]))
        V = tf.nn.conv2d(input_block, weight_init_G('channel_conv3'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        V = tf.nn.bias_add(V, bias_init_G('channel_bias3' + weight_name, [channel_num]))
        
        
        Q = tf.reshape(Q,[-1, w * h, channel_num])
        Q = tf.transpose(Q, [0,2,1])
        K = tf.reshape(K,[-1, w * h, channel_num])
        V = tf.reshape(V,[-1, w * h, channel_num])
        V = tf.transpose(V, [0,2,1])
        print('The shape of Q is',Q.get_shape())
        
        channel_weight = tf.nn.softmax(tf.matmul(Q,K))
        print('The shpae of channel_weight is', channel_weight.get_shape())
        
        weighted_block = tf.matmul(channel_weight,V)
        weighted_block = tf.transpose(weighted_block)
        weighted_block = tf.reshape(weighted_block, [-1, w, h, channel_num])
        print('The shape of weighted_block is',weighted_block.get_shape())
        
        channel_attation_block = (input_block + weighted_block)
        return channel_attation_block
    
    def channel_attation_x(self,input_block,weight_name):
        w = input_block.get_shape()[1]
        h = input_block.get_shape()[2]
        channel_num = input_block.get_shape()[3]
        
        Q = tf.nn.conv2d(input_block, weight_init('channel_conv1' + weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        Q = tf.nn.bias_add(Q, bias_init('channel_bias1' + weight_name, [channel_num]))
        K = tf.nn.conv2d(input_block, weight_init('channel_conv2'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        K = tf.nn.bias_add(K, bias_init('channel_bias2' + weight_name, [channel_num]))
        V = tf.nn.conv2d(input_block, weight_init('channel_conv3'+ weight_name,[1,1,channel_num,channel_num]), strides=[1, 1, 1, 1], padding='SAME')
        V = tf.nn.bias_add(V, bias_init('channel_bias3' + weight_name, [channel_num]))
        
        
        Q = tf.reshape(Q,[-1, w * h, channel_num])
        Q = tf.transpose(Q, [0,2,1])
        K = tf.reshape(K,[-1, w * h, channel_num])
        V = tf.reshape(V,[-1, w * h, channel_num])
        V = tf.transpose(V, [0,2,1])
        print('The shape of Q is',Q.get_shape())
        
        channel_weight = tf.nn.softmax(tf.matmul(Q,K))
        print('The shpae of channel_weight is', channel_weight.get_shape())
        
        weighted_block = tf.matmul(channel_weight,V)
        weighted_block = tf.transpose(weighted_block)
        weighted_block = tf.reshape(weighted_block, [-1, w, h, channel_num])
        print('The shape of weighted_block is',weighted_block.get_shape())
        
        channel_attation_block = (input_block + weighted_block)
        return channel_attation_block
    
    def weighted_spatial_and_channel(self, spatial_block, channel_block):
        spatial_block_weight = weight_init_G('spatial_block_weight',[1])
        channel_block_weight = weight_init_G('channel_block_weight',[1])
        output_block = spatial_block_weight * spatial_block + channel_block_weight * channel_block
        return output_block

    def HP_2_RGB(self,HPimg,filter_name):  
        HPimg = tf.nn.conv2d(HPimg, weight_init_D(filter_name,[1,1,31,3]), strides=[1, 1, 1, 1], padding='SAME')
        HPimg = tf.nn.bias_add(HPimg, bias_init_D(filter_name, [3]))
        return HPimg
    
    def MHINet_restore(self, ckpt_path):
        self.MHINet_saver.restore(self.sess, ckpt_path)
        
    def CVNet_restore(self, ckpt_path):
        self.CVNet_saver.restore(self.sess, ckpt_path)
        
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def test_externel(self):
        test_externel_datas = load_data(TEST_EXTERNEL_ITEMS)
        test_externel_data = [[], [], [], [], []]
        test_externel_count = 0
        test_externel_collection = []
        for r in test_externel_datas:
            test_externel_data[0].append(r.simulate_image)
            test_externel_data[1].append(r.ground_truth)
            test_externel_data[2].append(r.filename)
            test_externel_data[3].append(r.color_checker)
            test_externel_data[4].append(r.camera_response)
            test_externel_count += 1
            if test_externel_count >= BATCH_SIZE:
                test_externel_collection.append(test_externel_data) 
                test_externel_count = 0
                test_externel_data = [[], [], [], [], []]
        if test_externel_count != 0:
            test_externel_collection.append(test_externel_data)

        test_externel_loss_sum = 0
        test_externel_mat_count = 0
        test_externel_constructed_images_all = []
        test_externel_ground_truth_all = []
        test_externel_filenames_all = []
        test_externel_images_all = []
        test_externel_color_checker_all = []
        test_externel_camera_response_all = []
        test_externel_constructed_responses_all = []
        
        for test_externel_bag in test_externel_collection:
            training_loss_G, test_externel_loss, test_externel_constructed_images, constructed_response, achromatic_image = self.sess.run(
                [self.total_loss,
                 self.MSE_loss,
                 self.fake_image,
                 self.wave,
                 self.achromatic_image
                 ],
                feed_dict=
                {self.dropout: 1.0,
                 self.reconstructed_images: np.stack(test_externel_bag[0]),
                 self.ground_truth: np.stack(test_externel_bag[1]),
                 self.color_checker_image: np.stack(test_externel_bag[3]),
                 self.is_training : True
                 })
            if test_externel_mat_count == 0:
                #test_externel_constructed_images_all = test_externel_constructed_images
                test_externel_images_all = np.stack(test_externel_bag[0])
                test_externel_ground_truth_all = np.stack(test_externel_bag[1])
                test_externel_filenames_all = np.stack(achromatic_image)
                test_externel_color_checker_all = np.stack(test_externel_bag[3])
                test_externel_camera_response_all = np.stack(test_externel_bag[4])
                test_externel_constructed_responses_all = constructed_response
            else:
                #test_externel_constructed_images_all = np.concatenate([test_externel_constructed_images_all, test_externel_constructed_images], axis=0)
                test_externel_images_all = np.concatenate([test_externel_images_all, np.stack(test_externel_bag[0])], axis=0)
                test_externel_ground_truth_all = np.concatenate([test_externel_ground_truth_all, np.stack(test_externel_bag[1])], axis=0)
                test_externel_filenames_all = np.concatenate([test_externel_filenames_all, np.stack(achromatic_image)], axis=0)
                #test_externel_color_checker_all = np.concatenate([test_externel_color_checker_all, np.stack(test_externel_bag[3])], axis=0)
                #test_externel_camera_response_all = np.concatenate([test_externel_camera_response_all, np.stack(test_externel_bag[4])], axis=0)
                #test_externel_constructed_responses_all = np.concatenate([test_externel_constructed_responses_all, constructed_response], axis=0)
                
            test_externel_mat_count += 1
            test_externel_loss_sum = test_externel_loss_sum + test_externel_loss
        test_externel_loss_ave = test_externel_loss_sum / len(test_externel_collection)
        print('test: now the final test loss is: ', [test_externel_loss_ave, training_loss_G, len(test_externel_collection)])

        scipy.io.savemat('all_results/mat/test_external.mat', {'ground_truths': test_externel_ground_truth_all,
                                                      #'HSI': test_externel_constructed_images_all,
                                                      'constructed_images': test_externel_filenames_all,
                                                      #'color_checker': test_externel_color_checker_all,
                                                      #'camera_response': test_externel_camera_response_all,
                                                      #'constructed_responses': test_externel_constructed_responses_all,
                                                      'input_image': test_externel_images_all})

        
    def train(self, epoch_nums=EPOCH_NUMS):
        # self.train_epoch_writer_D = tf.summary.FileWriter('all_results/train_epoch_D/', graph=self.sess.graph)
        # self.train_batch_writer_D = tf.summary.FileWriter('all_results/train_batch_D/', graph=self.sess.graph)
        self.train_epoch_writer_G = tf.summary.FileWriter('all_results/train_epoch_G/', graph=self.sess.graph)
        self.train_batch_writer_G = tf.summary.FileWriter('all_results/train_batch_G/', graph=self.sess.graph)
        self.validation_writer = tf.summary.FileWriter('all_results/validation/', graph=self.sess.graph)
        
        self.tee = Tee('all_results/log.txt')
        validation_summary_input = tf.placeholder(tf.float32, shape=())
        validation_summary = tf.summary.scalar('loss_validation', validation_summary_input)
        
        batch_size = BATCH_SIZE
        print("Now train is running", TRAINING_CROSS_ITEMS)
        #trainning_data_generating = DataGenerator(True, TRAINING_CROSS_ITEMS, batch_size)
        batch_nums = len(TRAINING_CROSS_ITEMS) // batch_size
        print('The batch num is: ', batch_nums)
        tf.get_default_graph().finalize()

        validation_split_items = data_cross_split(VALIDATION_CROSS_ITEMS, 10, False)
        print(validation_split_items)
        for index, validation_item in enumerate(validation_split_items):
            validation_datas = load_data(validation_item)
            validation_data = [[], [], [], [], []]
            validation_count = 0
            validation_collection = []
            for r in validation_datas:
                validation_data[0].append(r.simulate_image)
                validation_data[1].append(r.ground_truth)
                validation_data[2].append(r.filename)
                validation_data[3].append(r.color_checker)
                validation_data[4].append(r.camera_response)
                validation_count += 1
                if validation_count >= BATCH_SIZE:
                    validation_collection.append(validation_data)
                    validation_count = 0
                    validation_data = [[], [], [], [], []]
            if validation_count != 0:
                validation_collection.append(validation_data)
    
            validation_mat_cnt = 0
            validation_ground_truth_all = []
            validation_filenames_all = []
            validation_images_all = []
            validation_color_checker_all = []
            validation_camera_response_all = []
            
            for validation_bag in validation_collection:
                if validation_mat_cnt == 0:
                    validation_images_all = np.stack(validation_bag[0])
                    validation_ground_truth_all = np.stack(validation_bag[1])
                    validation_filenames_all = np.stack(validation_bag[2])
                    validation_color_checker_all = np.stack(validation_bag[3])
                    validation_camera_response_all = np.stack(validation_bag[4])
                    
                else:
                    validation_images_all = np.concatenate([validation_images_all, np.stack(validation_bag[0])], axis=0)
                    validation_ground_truth_all = np.concatenate([validation_ground_truth_all, np.stack(validation_bag[1])],axis=0)
                    validation_filenames_all = np.concatenate([validation_filenames_all, np.stack(validation_bag[2])],axis=0)
                    validation_color_checker_all = np.concatenate([validation_color_checker_all, np.stack(validation_bag[3])],axis=0)
                    validation_camera_response_all = np.concatenate([validation_camera_response_all, np.stack(validation_bag[4])],axis=0)
                                     
                validation_mat_cnt += 1
            
            scipy.io.savemat('all_results/mat/validation_groundtruth_' + str(index) +'.mat',
                             {'ground_truths': validation_ground_truth_all, 
                              'filenames': validation_filenames_all,
                              'color_checker': validation_color_checker_all,
                              'camera_response': validation_camera_response_all})

        validation_loss_best = 0
        training_loss_last = 0
        scalar_summaries_validation = []

        for i in range(1+INTERUPT_INDEX, EPOCH_NUMS+1) :                      
            training_loss_sum_D = 0
            training_loss_sum_G = 0
            #TRAINING_CROSS_ITEMS.shuffle()
            # print('Now the learning rate is :', self.sess.run(self.learning_rate))
            # print('')
            batch_items = One_batch_items(TRAINING_CROSS_ITEMS, batch_size)
            for batch in range(batch_nums):
                #one_batch = trainning_data_generating.get_one_batch()              
                one_batch_items = batch_items.get_one_batch_items()
                one_batch_collection = load_data(one_batch_items)
                one_batch = [[], [], [], [], []]
                for x in one_batch_collection:
                    one_batch[0].append(x.simulate_image)
                    one_batch[1].append(x.ground_truth)
                    one_batch[2].append(x.filename)
                    one_batch[3].append(x.color_checker)
                    one_batch[4].append(x.camera_response) 
                edit_images_fake, edit_images_real = self.sess.run([self.edit_images_fake, self.edit_images_real], feed_dict={self.raw_images_fake: np.stack(one_batch[0]), self.raw_images_real: np.stack(one_batch[1])})
                training_loss_G, _, training_summary_G, MSE_loss, percep_loss, phase_loss = self.sess.run(
                    [self.loss,
                     self.train_step_adam_G,
                     self.scalar_summaries_G,
                     self.MSE_loss,
                     self.percep_loss,
                     self.phase_MSE_loss
                     ],
                    feed_dict=
                    {self.dropout: DROPOUT,
                     self.learning_rate: LEARNING_RATE,
                     self.reconstructed_images: np.stack(edit_images_fake),
                     self.ground_truth: np.stack(edit_images_real),
                     self.color_checker_image: np.stack(one_batch[3]),
                     self.camera_response : np.stack(one_batch[4]),
                     self.is_training : True
                     })
                training_loss_sum_G += training_loss_G
                #print('The' , i, 'G_epoch', batch, 'batch',' training loss is ', ['D_loss','G_loss','MSE_loss','G_loss_fake'], [training_loss_D,training_loss_G,MSE_loss,G_loss_fake])
                print('The', i , 'epoch', batch, 'batch',' training loss is ',[training_loss_G,percep_loss,MSE_loss,phase_loss])
                self.train_batch_writer_G.add_summary(training_summary_G, (i-1) * batch_nums + batch)
                training_loss_last_G = training_loss_G
            
            training_loss_ave_G = training_loss_sum_G / batch_nums
            self.train_epoch_writer_G.add_summary(training_summary_G, i)
            print('Training: the ',i, ' epoch training loss_ave is: ', training_loss_ave_G)
            print('')
             
            #scalar_summaries_validation.append(tf.summary.scalar('validation_loss', training_loss_ave_G))
            #training_summaries_validation = tf.summary.merge(scalar_summaries_validation)
            #validation_summary
            validation_total_loss = 0
            validation_split_items = data_cross_split(VALIDATION_CROSS_ITEMS, 10, False)
            for validation_external_index, validation_item in enumerate(validation_split_items):
                validation_datas = load_data(validation_item)
                validation_data = [[], [], [], [], []]
                validation_count = 0
                validation_collection = []
                for r in validation_datas:
                    validation_data[0].append(r.simulate_image)
                    validation_data[1].append(r.ground_truth)
                    validation_data[2].append(r.filename)
                    validation_data[3].append(r.color_checker)
                    validation_data[4].append(r.camera_response)
                    validation_count += 1
                    if validation_count >= BATCH_SIZE:
                        validation_collection.append(validation_data)
                        validation_count = 0
                        validation_data = [[], [], [], [], []]
                if validation_count != 0:
                    validation_collection.append(validation_data)
    
                validation_loss_sum = 0
                validation_MSE_loss_sum = 0
                validation_percep_loss_sum = 0
                validation_phase_loss_sum = 0
                validation_constructed_images_all = []
                validation_constructed_responses_all = []
                validation_input_images_all = []
                validation_spectral_images_all = []
                validation_mat_count = 0
                for validation_bag in validation_collection:
                    edit_images_fake, edit_images_real = self.sess.run([self.edit_images_fake, self.edit_images_real], feed_dict={self.raw_images_fake: np.stack(validation_bag[0]), self.raw_images_real: np.stack(validation_bag[1])})
                    validation_loss, _, MSE_loss, percep_loss, validation_constructed_images, validation_constructed_response, phase_loss, lens, psf, HSI = self.sess.run(
                        [self.loss,
                         self.train_step_adam_G,
                         self.MSE_loss,
                         self.percep_loss,
                         self.achromatic_image,
                         self.recovered_pahse,
                         self.phase_MSE_loss,
                         self.lens,
                         self.psf,
                         self.spectral_image
                         ],
                        feed_dict=
                        {self.dropout: 1.0,
                         self.learning_rate: VALIDATION_LEARNING_RATE,
                         self.reconstructed_images: np.stack(edit_images_fake),
                         self.ground_truth: np.stack(edit_images_real),
                         self.color_checker_image: np.stack(one_batch[3]),
                         self.camera_response : np.stack(one_batch[4]),
                         self.is_training : True
                         })
                    # training_loss_G, SSIM_loss, COS_loss, validation_loss, validation_constructed_images, validation_constructed_response = self.sess.run(
                    #     [self.total_G_loss,
                    #      self.SSIM_loss,
                    #      self.COS_loss,
                    #      self.MSE_loss,
                    #      self.fake_image,
                    #      self.constructed_response
                    #      ],
                    #     feed_dict=
                    #     {self.dropout: 1.0,
                    #      self.reconstructed_images: np.stack(validation_bag[0]),
                    #      self.ground_truth: np.stack(validation_bag[1]),
                    #      self.color_checker_image: np.stack(validation_bag[3]),
                    #      self.camera_response: np.stack(validation_bag[4]),
                    #      self.is_training : True
                    #      })
                    validation_loss_sum += validation_loss
                    validation_MSE_loss_sum += MSE_loss
                    validation_percep_loss_sum += percep_loss
                    validation_phase_loss_sum += phase_loss
                    if validation_mat_count == 0:
                        validation_constructed_images_all = validation_constructed_images
                        validation_constructed_responses_all = validation_constructed_response
                        validation_input_images_all = np.stack(edit_images_fake)
                        validation_spectral_images_all = HSI
                    else:
                        validation_constructed_images_all = np.concatenate(
                            [validation_constructed_images_all, validation_constructed_images], axis=0)
                        validation_constructed_responses_all = np.concatenate(
                            [validation_constructed_responses_all, validation_constructed_response], axis=0)
                        validation_input_images_all = np.concatenate(
                            [validation_input_images_all, np.stack(edit_images_fake)], axis=0)
                        validation_spectral_images_all = np.concatenate(
                            [validation_spectral_images_all, HSI], axis=0)
                    validation_mat_count += 1
    
                validation_loss_ave = validation_loss_sum / len(validation_collection)
                validation_MSE_loss_ave = validation_MSE_loss_sum / len(validation_collection)
                validation_percep_loss_ave = validation_percep_loss_sum/ len(validation_collection)
                validation_phase_loss_ave = validation_phase_loss_sum/ len(validation_collection)
                self.validation_writer.add_summary(
                    validation_summary.eval(feed_dict={validation_summary_input: validation_loss_ave}), i)
                #self.train_epoch_writer.add_summary(training_summary, i)
                print('Validation: ',i,'_',validation_external_index,' epoch loss is: ', [validation_loss_ave,
                                                                                          validation_MSE_loss_ave,
                                                                                          validation_percep_loss_ave,
                                                                                          validation_phase_loss_ave])
                # validation_loss_ave = validation_loss_sum / len(validation_collection)
                # validation_total_loss += validation_loss_ave
                # print('Validation: now the ', i, '_', validation_external_index, '_Epoch validation loss is: ', [validation_loss, SSIM_loss, COS_loss, validation_loss_ave])
                # print('')
                if (i-1) % 30 == 0:
                    scipy.io.savemat('all_results/mat/validation_' + str(i) + '_' + str(validation_external_index) + '.mat',{'constructed_images': validation_constructed_images_all,
                                                                                                                             'constructed_responses': validation_constructed_responses_all,
                                                                                                                             'input_images': validation_input_images_all,
                                                                                                                             'HSI':validation_spectral_images_all,
                                                                                                                             'lens' : lens,
                                                                                                                             'psf' : psf})

            #validation_total_loss_ave = validation_total_loss/len(validation_split_items)
            #self.validation_writer.add_summary(training_summaries_validation , i)
            #self.validation_writer.add_summary(validation_summary.eval(feed_dict={validation_summary_input: validation_total_loss_ave}), i)               
            
            if i == (1+INTERUPT_INDEX):
                validation_loss_best = validation_loss_ave
                ckpt_path = 'all_results/ckpt_result/' + str(i) + '.ckpt'
                self.saver.save(self.sess, ckpt_path)
            elif validation_loss_best > validation_loss_ave:
                validation_loss_best = validation_loss_ave
                ckpt_path = 'all_results/ckpt_result/' + str(i) + '.ckpt'
                self.saver.save(self.sess, ckpt_path)
        
        test_split_items = data_cross_split(TEST_CROSS_ITEMS, 10, False)
        for test_index,test_item in enumerate(test_split_items):
            test_datas = load_data(test_item)
            test_data = [[], [], [], [], []]
            test_count = 0
            test_collection = []
            for r in test_datas:
                test_data[0].append(r.simulate_image)
                test_data[1].append(r.ground_truth)
                test_data[2].append(r.filename)
                test_data[3].append(r.color_checker)
                test_data[4].append(r.camera_response)
                test_count += 1
                if test_count >= BATCH_SIZE:
                    test_collection.append(test_data) 
                    test_count = 0
                    test_data = [[], [], [], [], []]
            if test_count != 0:
                test_collection.append(test_data)
            
            test_total_MSE_loss = 0
            test_total_SSIM_loss = 0
            test_total_COS_loss = 0
            test_total_G_loss = 0
            test_mat_count = 0
            test_constructed_images_all = []
            test_ground_truth_all = []
            test_filenames_all = []
            test_images_all = []
            test_color_checker_all = []
            test_camera_response_all = []
            test_constructed_responses_all = []
            
            for test_bag in test_collection:
                training_loss_G, SSIM_loss, COS_loss, test_loss, test_constructed_images, test_constructed_response = self.sess.run(
                    [self.total_G_loss,
                     self.SSIM_loss,
                     self.COS_loss,
                     self.MSE_loss,
                     self.fake_image,
                     self.constructed_response
                     ],
                    feed_dict=
                    {self.dropout: 1.0,
                     self.reconstructed_images: np.stack(test_bag[0]),
                     self.ground_truth: np.stack(test_bag[1]),
                     self.color_checker_image: np.stack(test_bag[3]),
                     self.camera_response: np.stack(test_bag[4]),
                     self.is_training : False
                     })
                if test_mat_count == 0:
                    test_constructed_images_all = test_constructed_images
                    test_images_all = np.stack(test_bag[0])
                    test_ground_truth_all = np.stack(test_bag[1])
                    test_filenames_all = np.stack(test_bag[2])
                    test_color_checker_all = np.stack(test_bag[3])
                    test_camera_response_all = np.stack(test_bag[4])
                    test_constructed_responses_all = test_constructed_response
                else:
                    test_constructed_images_all = np.concatenate([test_constructed_images_all, test_constructed_images], axis=0)
                    test_images_all = np.concatenate([test_images_all, np.stack(test_bag[0])], axis=0)
                    test_ground_truth_all = np.concatenate([test_ground_truth_all, np.stack(test_bag[1])], axis=0)
                    test_filenames_all = np.concatenate([test_filenames_all, np.stack(test_bag[2])], axis=0)
                    test_color_checker_all = np.concatenate([test_color_checker_all, np.stack(test_bag[3])], axis=0)
                    test_camera_response_all = np.concatenate([test_camera_response_all, np.stack(test_bag[4])], axis=0)
                    test_constructed_responses_all = np.concatenate([test_constructed_responses_all, test_constructed_response], axis=0)
                    
                test_mat_count += 1
                test_total_MSE_loss = test_total_MSE_loss + test_loss
                test_total_SSIM_loss = test_total_SSIM_loss + SSIM_loss
                test_total_COS_loss = test_total_COS_loss + COS_loss
                test_total_G_loss = test_total_G_loss + training_loss_G
                
            test_SSIM_loss_ave = test_total_SSIM_loss / len(test_collection)
            test_MSE_loss_ave = test_total_MSE_loss / len(test_collection)
            test_COS_loss_ave = test_total_COS_loss / len(test_collection)
            test_G_loss_ave = test_total_G_loss / len(test_collection)
            print('test: now the final test loss is: [Total_loss, SSIM_loss, COS_loss, MSE_loss], ', [test_G_loss_ave, test_SSIM_loss_ave, test_COS_loss_ave, test_MSE_loss_ave])
    
            scipy.io.savemat('all_results/mat/test_' + str(test_index) +'.mat', {'ground_truths': test_ground_truth_all,
                                                          'constructed_images': test_constructed_images_all,
                                                          'filenames': test_filenames_all, 
                                                          'color_checker': test_color_checker_all,
                                                          'camera_response': test_camera_response_all,
                                                          'constructed_responses': test_constructed_responses_all,
                                                          'input_image': test_images_all})

