#!/usr/bin/python
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.inf)
import scipy
import cPickle as pickle
import time
import sys
sys.path.append('../py')
sys.path.append('../py/Piotr_Ozimek_retina')
import retina_cuda
import cortex_cuda
import retina
import cortex
import perf_test

retina_path = '../Retinas'
mat_data = '../Retinas'
coeff = [0,0,0,0,0]
loc = [0,0,0,0,0]

coeff[0] = scipy.io.loadmat(mat_data+'/coefficients.mat')['M']
coeff[1] = scipy.io.loadmat(mat_data+'/coeff4k.mat')['M4k']
coeff[2] = scipy.io.loadmat(mat_data+'/coeff1k.mat')['M1k']
coeff[3] = scipy.io.loadmat(retina_path+'/coeffv2_1.mat')['coeffv2']
loc[0] = scipy.io.loadmat(mat_data+'/locations.mat')['ind']
loc[1] = scipy.io.loadmat(mat_data+'/loc4k.mat')['ind4k']
loc[2] = scipy.io.loadmat(mat_data+'/loc1k.mat')['ind1k']
loc[3] = scipy.io.loadmat(retina_path+'/locv2_1.mat')['locv2']
with open(retina_path + '/ret50k_loc.pkl', 'rb') as handle:
    loc[4] = pickle.load(handle)
with open(retina_path + '/ret50k_coeff.pkl', 'rb') as handle:
    coeff[4] = pickle.load(handle)

img_sizes = [(480,320),(640,480),(800,600),(1080,720),(1280,1024),(1920,1080)]

# measure initialisation time, performance and Python difference
# for i in range(0,5):
#     for size in img_sizes[1:]:
#         perf_test.speedup_cam(loc[i],coeff[i],size, 20, True, False)

# measure raw performance in ideal envrionment (e.g. one initalisation, use object on a lot of data)
for i in range(0,5):
    for size in img_sizes:
        perf_test.ideal_usage_cam(loc[i],coeff[i],size, 20, True, False)