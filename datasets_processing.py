# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import scipy.io
import os
import random
import math
import numpy as np
import sys

class TrainingData(): 
    '''创建一个类，用实例接收时赋给实例3个对应的属性（ground_truth等）'''
    def __init__(self, filename, ground_truth, simulate_image, color_checker,camera_response):
        self.filename = filename
        self.ground_truth = ground_truth
        self.simulate_image = simulate_image
        self.color_checker = color_checker
        self.camera_response = camera_response

class SpectralData():
    '''光谱数据类，导入光谱反射率等'''
    def __init__(self, spec_direction):
        self.spec_names = sorted(os.listdir(spec_direction)) #将文件夹内文件按数字大小排序，并将文件名输出为列表'''
        self.spectral_datas = []
        self.spec_direction = spec_direction #导入数据的路径'''
        
    def load(self,record_dir):   
        '''载入数据，用空集合datas接收生成的数据''' 
        load_count=0
        for mat_item,name in enumerate(self.spec_names): #遍历所有mat文件'''
            load_count+=1
            if load_count%1000 == 0: #计数'''
                print('now we are loading ', load_count, ' data')
            matlab_data = scipy.io.loadmat(os.path.join(self.spec_direction, name)) #载入对应mat文件'''
            filename = matlab_data['filename']
            ground_truth = matlab_data['ground_truth_image']
            simulate_image = matlab_data['simulate_image']
            simulate_image = np.clip(simulate_image / simulate_image.max(), 0, 1)
            simulate_image = simulate_image.astype(np.float32)
            color_checker = matlab_data['color_checker']
            camera_response =  matlab_data['camera_response']   #将每一个mat的[filename,ground_truth,image取出后打包给item]
            item = TrainingData(filename[0], ground_truth, simulate_image,color_checker,camera_response) 
            
            record_dir_item = record_dir + str(mat_item) + '.pkl'
            print('the ',mat_item + 1,' of', len(self.spec_names))        
            data_dump(item, record_dir_item)
            
            
def cross_split(datas, slice_nums):#数据切块儿'''
    datas_cross = [[][:] for index in range(slice_nums)] #生成一个空集合和对应的索引index'''
    random.shuffle(datas) #随机打乱数据'''
    number = math.ceil(len(datas)/slice_nums)
    for i in range(slice_nums):
        print('\n we are spliting ', i, ' datas')
        start = i * number
        end = min(start + number, len(datas)) 
        for j in range(start, end):           #区间长度为切片数量'''
            datas_cross[i].append(datas[j])
            print('the ', j , ' filename is', datas[j].filename)
            
    return datas_cross
    
def data_dump(datas_cross, record_dir):
    with open(record_dir, 'wb') as file_object:
        pickle.dump(datas_cross, file_object, protocol=-1)     #将data_cross写入file_object'''
    
    
def train_make(): #'''训练数据生成''''
    spec_dir = 'mat_data'#数据保存路径'''
    record_dir = 'training_data/dump_training_data'
    spectral_data = SpectralData(spec_dir)
    print('data was dumped now')
    spectral_data.load(record_dir)
    #training_datas_cross = cross_split(training_datas, 10)   #生成数据后切片'''
    print('dump over')
        
def test_make(): #测试数据生成，同上'''
    spec_dir = 'test_external_spctral_data/mat_data'
    record_dir = 'test_external_spctral_data/dump_data' + '/dumped_spectra_data_'
    test_datas = []
    spectral_data = SpectralData(spec_dir)
    spectral_data.load(test_datas)
    test_datas_cross = cross_split(test_datas, 1)
    print('data was dumped now')
    for item, cross_data in enumerate(test_datas_cross):
        record_dir_item = record_dir + '99.pkl'
        print('the ',item + 1,' of', len(test_datas_cross), ' training datas is', len(cross_data))            
        data_dump(cross_data, record_dir_item)    
    print('dump over')
    


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please tell me more about the function you want, such as train_make or test_make')
        exit(-1)
    function = sys.argv[1]
    globals()[function](*sys.argv[2:])

    
    