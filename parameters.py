# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:48:50 2020

@author: Administrator
"""
import scipy.io
import numpy as np

EPOCH_NUMS = 1000

INTERUPT_INDEX = 0

WEIGHT_DECAY = 1e-5

MAX_TO_KEEP = 1

RAW_IMAGE_SIZE = 716

PHASE_SIZE = 716

IMAGE_SIZE = 716

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

VALIDATION_LEARNING_RATE = 0.0

DROPOUT = 1.0

TRAINING_CROSS_ITEMS = list(range(0,450))

VALIDATION_CROSS_ITEMS = list(range(450,480))

TEST_CROSS_ITEMS = list(range(480,500))

TEST_EXTERNEL_ITEMS =  list(range(0,1))

'''Enable'''
PREPROCESS_ENABLE = False

RANDOM_FLIP_LR_ENABLE = False

RANDOM_FLIP_UD_ENABLE = False

RANDOM_ROTATE_ENABLE = False

RANDOM_CROP_ENABLE = False

ROTATE_MAX_ANGLE = 45

CROP_LEAST_COVER_PERCENTAGE = 0.8

CHANNELS = 381

PIC_SHOW = False

GAMMA_ENABLE = False

GENERATOR_ENABLE = True

OPTIC_ENABLE = True

BUILD_ENABLE = True

OPTICAL_MODEL_TRAINING_ENABLE = False

BASIC_FUNCTION_MAT = scipy.io.loadmat('basic_fun/fft=16.mat')

BASIC_FUNCTIONS = BASIC_FUNCTION_MAT['y']

BASIC_NUMS = BASIC_FUNCTIONS.shape[1]

''''optics parameters'''

# wave_lengths = np.array(list(range(400, 700, 30))) * 1e-9

# refractive_idcs = (550 * 1.493)/(np.array(list(range(400, 700, 30))))

PHASE_NUM = 36

spectral_num = 10

complex_num = 8

wave_list = []

refractive_list = []
for i in range(0, spectral_num):
    for j in range(0, complex_num):
        wave_list.append(400+i*(300//spectral_num))
        refractive_list.append((550 * 1.493)/(400+i*(300//spectral_num)))
        
wave_lengths = np.array(wave_list) * 1e-9

refractive_idcs = (np.array(refractive_list))

patch_size = 512

sampling_interval = 5.86e-6  # 5.86e-6 # pixel size /m

wave_resolution = 853, 853

sensor_distance = 87.36e-3

obj_distance = 50e-3

focal_length = [31.8e-3]*(spectral_num*complex_num)
