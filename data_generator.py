# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:23:01 2020

@author: Administrator
"""
import pickle
import random
from parameters import *
from datasets_processing import TrainingData
import tensorflow as tf
import matplotlib.pyplot as plt
#import cv2
import math
import time


def load_data(items):
    data = []
    num = 1
    for item in items:
        begin = time.time()
        data_item_dir = './test_pkl/' + str(item) +'.pkl'
        with open(data_item_dir, 'rb') as file_object:
            data = data + [pickle.load(file_object)]
            #print('Now', num, 'pkl is loaded')
            num +=1
        end = time.time()
        timer = end - begin
        #print('Loading time is :', timer/60)
    return data


# def load_data(items):
#     data = []
#     for item in items:
#         data_item_dir = '../training_data/dump_training_data_1024_&_512/' + str(item) +'.pkl'
#         with open(data_item_dir, 'rb') as file_object:
#             data = data + [pickle.load(file_object)]
#     return data
# def load_data(items):
#     print('The item is: ',items)
#     data = []
#     num = 0
#     for item in items:
#         print('Now the item is: ',item)
#         data_item_dir = '../training_data/dump_training_data_1024/' + str(item) +'.pkl'
#         with open(data_item_dir, 'rb') as file_object:
#             data = data + pickle.load(file_object)
#             print('Now', num, 'pkl is loaded')
#             num +=1
#     return data

def preprocess(image):
    if PREPROCESS_ENABLE:
        height = image.shape[0]
        width = image.shape[1]
        if RANDOM_FLIP_LR_ENABLE:
            image = tf.image.random_flip_left_right(image)
        if RANDOM_FLIP_UD_ENABLE:
            image = tf.image.random_flip_up_down(image)
        if RANDOM_ROTATE_ENABLE:
            image = tf.contrib.image.rotate(image, (random.random() - 0.5) * math.radians(ROTATE_MAX_ANGLE))
        if RANDOM_CROP_ENABLE:
            xmax = 1 - random.random()*(1 - CROP_LEAST_COVER_PERCENTAGE)
            ymax = 1 - random.random() *(1 - CROP_LEAST_COVER_PERCENTAGE)
            start = max(0, xmax - ymax * height / width)
            end = ymax - CROP_LEAST_COVER_PERCENTAGE
            xmin = start + random.random() * (end - start)
            ymin = ymax - (xmax - xmin) * width / height
            image = tf.expand_dims(image, 0)       
            image = tf.image.crop_and_resize(image,[[ymin,xmin,ymax,xmax]],box_ind=[0],crop_size=(512,512))            
            
    return image

class DataGenerator:
    '''训练数据生成''' 
    def __init__(self, is_training, items, batch_size):
        self.is_training = is_training
        self.data = load_data(items)
        random.shuffle(self.data) 
        '''随机打乱数据'''
        self.batch_size = batch_size
        self.data_nums = len(self.data)
        self.count = 0
        self.ground_truth = []
        self.simulate_image = []  
        self.filename = []
        self.color_checker = []
        self.camera_response = []
        for i in range(self.data_nums):   
            '''' [ [self.ground],[self.simulate_image],[self.filename] ] '''
            self.ground_truth.append(self.data[i].ground_truth)
            self.simulate_image.append(self.data[i].simulate_image)
            self.filename.append(self.data[i].filename)
            self.color_checker.append(self.data[i].color_checker)
            self.camera_response.append(self.data[i].camera_response)
        
    def get_one_batch(self):
        '''获取一个batch的数据'''
        one_batch = [[], [], [], [], []]
        if self.count + self.batch_size > self.data_nums:
            self.count = 0
            #random.shuffle(self.data) 
            '''取完一轮后随机打乱'''
        for i in range(self.count, self.count + self.batch_size):
            one_batch[0].append(self.simulate_image[i])
            one_batch[1].append(self.ground_truth[i])
            one_batch[2].append(self.filename[i])
            one_batch[3].append(self.color_checker[i])
            one_batch[4].append(self.camera_response[i])
            
        self.count = self.count + self.batch_size
        return one_batch
        
class One_batch_items:
    
    def __init__(self, items_list, batch_size):
        self.items_list = items_list
        self.batch_size = batch_size
        self.count = 0
        
    def get_one_batch_items(self):
        if self.count + self.batch_size > len(self.items_list):
            self.count = 0
            #random.shuffle(self.items_list)
        items = self.items_list[self.count : (self.count + self.batch_size)]
        self.count = self.count + self.batch_size
        
        return items
        
        
        
        
        
        
        
        
        
        
        
        
         
                
            
