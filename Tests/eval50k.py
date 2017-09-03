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

retina_path = '../Retinas'
mat_data = '../Retinas'

with open(retina_path + '/ret50k_loc.pkl', 'rb') as handle:
    loc50k = pickle.load(handle)
with open(retina_path + '/ret50k_coeff.pkl', 'rb') as handle:
    coeff50k = pickle.load(handle)

img_sizes = [(480,320),(640,480),(800,600),(1080,720),(1280,1024),(1920,1080)]

if __name__ == '__main__':
    args = sys.argv
    print 'eval 50k'
    print args
    
    rgb = args[1] == 'True'
    img_ind = int(args[2])
    stop = int(args[3])
    show_res = args[4] == 'True'
    camid = -1
    cap = cv2.VideoCapture(camid)

    while not cap.isOpened():
        print 'retrying\n'
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1

    retinas = np.empty((2,5), dtype=object)
    cortexes = np.empty((2,5), dtype=object)

    r, img = cap.read()
    while not r: r, img = cap.read()

    img = cv2.resize(img, img_sizes[img_ind])
    L, R = cortex.LRsplit(loc50k)
    L_loc, R_loc = cortex.cort_map(L, R)
    L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)

    if rgb: ret = retina_cuda.create_retina(loc50k, coeff50k, img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
    else: ret = retina_cuda.create_retina(loc50k, coeff50k, (img.shape[0], img.shape[1]), (int(img.shape[1]/2), int(img.shape[0]/2)))
    cort = cortex_cuda.create_cortex_from_fields_and_locs(L, R, L_loc, R_loc, cort_size, gauss100=G, rgb=rgb)
    count = 0
    while count < stop:
        r, img = cap.read()
        img = cv2.resize(img, img_sizes[img_ind])
        if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if r:
            start = time.time()
            V_c = ret.sample(img) # sample with CUDA
            sample = time.time()
            inv_c = ret.inverse(V_c) # inverse with CUDA
            inv = time.time()
            l_c = cort.cort_image_left(V_c) # left cortical image CUDA
            r_c = cort.cort_image_right(V_c) # right cortical image CUDA
            end = time.time()
            count += 1
        print '%f,%f,%f,' % (sample-start, inv-sample, end-inv)
        if show_res:
            c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1)

            cv2.namedWindow("inverse CUDA", cv2.WINDOW_NORMAL)
            cv2.imshow("inverse CUDA", inv_c)
            cv2.namedWindow("cortex CUDA", cv2.WINDOW_NORMAL)
            cv2.imshow("cortex CUDA", c_c)
            cv2.waitKey(10)
