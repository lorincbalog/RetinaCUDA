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
import os
import retina_cuda
import cortex_cuda
import retina
import cortex

retina_path = '../Retinas'
mat_data = '../Retinas'
coeff = [0,0,0,0]
loc = [0,0,0,0]

coeff[0] = scipy.io.loadmat(mat_data+'/coefficients.mat')['M']
coeff[1] = scipy.io.loadmat(mat_data+'/coeff4k.mat')['M4k']
coeff[2] = scipy.io.loadmat(mat_data+'/coeff1k.mat')['M1k']
coeff[3] = scipy.io.loadmat(retina_path+'/coeffv2_1.mat')['coeffv2']
loc[0] = scipy.io.loadmat(mat_data+'/locations.mat')['ind']
loc[1] = scipy.io.loadmat(mat_data+'/loc4k.mat')['ind4k']
loc[2] = scipy.io.loadmat(mat_data+'/loc1k.mat')['ind1k']
loc[3] = scipy.io.loadmat(retina_path+'/locv2_1.mat')['locv2']
ret50k = scipy.io.loadmat(retina_path+'/ret50k_sorted.mat')
with open(retina_path + '/ret50k_loc.pkl', 'rb') as handle:
    loc50k = pickle.load(handle)
with open(retina_path + '/ret50k_coeff.pkl', 'rb') as handle:
    coeff50k = pickle.load(handle)

def speedup(loc, coeff, img, rgb, show_res):
    '''
    This test measures the performance of the two implementation
    from initialisation to the end of the cortical transform
    '''
    init_p = time.time()
    GI = retina.gauss_norm_img(int(img.shape[1]/2), int(img.shape[0]/2), coeff, loc, img.shape, rgb)
    
    init_c = time.time()
    ret = retina_cuda.create_retina(loc, coeff, img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
    
    sample_p = time.time()
    V_p = retina.sample(img, img.shape[1]/2, img.shape[0]/2, coeff, loc, rgb)
    
    sample_c = time.time()
    V_c = ret.sample(img)
    
    invert_p = time.time()
    inv_p = retina.inverse(V_p, img.shape[1]/2, img.shape[0]/2, coeff, loc, GI, img.shape, rgb)
    
    invert_c = time.time()
    inv_c = ret.inverse(V_c)
    retina_end = time.time()

    cort_init_p = time.time()
    L, R = cortex.LRsplit(loc)
    L_loc, R_loc = cortex.cort_map(L, R)
    L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)

    cort_init_c = time.time()
    cort = cortex_cuda.create_cortex_from_fields(loc, rgb=rgb)

    cort_img_p = time.time()
    l_p, r_p = cortex.cort_img(V_p, L, L_loc, R, R_loc, cort_size, G)

    cort_img_c = time.time()    
    l_c = cort.cort_image_left(V_c)
    r_c = cort.cort_image_right(V_c)
    cort_end = time.time()

    print '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,' % (init_c - init_p, sample_p - init_c, sample_c - sample_p, \
                                  invert_p - sample_c, invert_c - invert_p, retina_end - invert_c,\
                                  cort_init_c - cort_init_p, cort_img_p - cort_init_c, cort_img_c - cort_img_p, cort_end - cort_img_c)

    if show_res:
        cv2.namedWindow("inverse CUDA", cv2.WINDOW_NORMAL)
        cv2.imshow("inverse CUDA", inv_c)
        cv2.namedWindow("inverse Piotr", cv2.WINDOW_NORMAL)
        cv2.imshow("inverse Piotr", inv_p)
        c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1)
        c_p = np.concatenate((np.rot90(l_p),np.rot90(r_p,k=3)),axis=1)
        cv2.namedWindow("cortex CUDA", cv2.WINDOW_NORMAL)
        cv2.imshow("cortex CUDA", c_c)
        cv2.namedWindow("cortex Piotr", cv2.WINDOW_NORMAL)
        cv2.imshow("cortex Piotr", c_p)

def speedup_cam(loc, coeff, img_size, stop=-1, show_res=False, rgb=False):
    camid = -1
    cap = cv2.VideoCapture(camid)

    while not cap.isOpened():
        print 'retrying\n'
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1
    count = 0
    while cv2.waitKey(10) != ord('q'):
        r, img = cap.read()
        if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if r:
            img = cv2.resize(img, img_size)
            speedup(loc, coeff, img, rgb, show_res)
            count += 1
        if count == stop: break

def ideal_usage_cam(loc, coeff, img_size, stop=-1, show_res=False, rgb=False):
    '''
    This test shows how the library ideally should be used (unchanged parameters)
    and measures the performance of this best-case scenario
    '''
    camid = -1
    cap = cv2.VideoCapture(camid)

    while not cap.isOpened():
        print 'retrying\n'
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1
    
    r, img = cap.read()
    if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # instantiate a retina
    img = cv2.resize(img, img_size)
    ret = retina_cuda.create_retina(loc, coeff, img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
    # instantiate a cortex
    cort = cortex_cuda.create_cortex_from_fields(loc, rgb=rgb)
    # for best performance, do not change these objects

    count = 0
    while cv2.waitKey(10) != ord('q'):
        r, img = cap.read()
        if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, img_size)
        if r:
            sample = time.time()
            V = ret.sample(img)

            invert = time.time()
            inv_c = ret.inverse(V)

            cortical = time.time()
            l_c = cort.cort_image_left(V)
            r_c = cort.cort_image_right(V)
            
            end = time.time()

            print ('%f,%f,%f,' % (invert - sample, cortical - invert, end - cortical))
            count += 1
            if show_res:
                cv2.namedWindow("inverse CUDA", cv2.WINDOW_NORMAL)
                cv2.imshow("inverse CUDA", inv_c)
                c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1)
                cv2.namedWindow("cortex CUDA", cv2.WINDOW_NORMAL)
                cv2.imshow("cortex CUDA", c_c)
        if count == stop : break
                
if __name__ == "__main__":
    #ideal_usage_cam(loc[0], coeff[0], True, False)
    speedup_cam(loc[0], coeff[0], (640,480), show_res=True, rgb=True)
    #speedup_cam(loc[0], coeff[0], True, False)
    #speedup_cam(loc[1], coeff[1], True, True)
    #speedup_cam(loc[1], coeff[1], True, False)
    
    quit()