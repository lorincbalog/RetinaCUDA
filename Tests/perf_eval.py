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
import subprocess

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

if __name__ == '__main__':
    args = sys.argv
    print args

    func = args[1]
    ind = int(args[2])
    size_ind = int(args[3])
    stop = int(args[4])
    show_res = (args[5] == 'True')
    color = (args[6] == 'True')
    if func == 'speedup':
        perf_test.speedup_cam(loc[ind],coeff[ind], img_sizes[size_ind], stop, show_res, color)
    else:
        perf_test.ideal_usage_cam(loc[ind],coeff[ind], img_sizes[size_ind], stop, show_res, color)