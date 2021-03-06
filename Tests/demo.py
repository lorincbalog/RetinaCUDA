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
with open(retina_path + '/ret50k_loc.pkl', 'rb') as handle:
    loc50k = pickle.load(handle)
with open(retina_path + '/ret50k_coeff.pkl', 'rb') as handle:
    coeff50k = pickle.load(handle)

img_sizes = [(480,320),(640,480),(800,600),(1080,720),(1280,1024),(1920,1080)]

L, R = cortex.LRsplit(loc50k)
L_loc, R_loc = cortex.cort_map(L, R)
L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)

retinas = np.empty((2,5,6), dtype=object)
cortexes = np.empty((2,5), dtype=object)

for i in range(0,4):
    for ind, img in enumerate(img_sizes):
        retinas[0,i,ind] = retina_cuda.create_retina(loc[i], coeff[i], (img[1],img[0],3), (int(img[0]/2), int(img[1]/2)))
        retinas[1,i,ind] = retina_cuda.create_retina(loc[i], coeff[i], (img[1], img[0]), (int(img[0]/2), int(img[1]/2)))
    cortexes[0,i] = cortex_cuda.create_cortex_from_fields(loc[i], rgb=True)
    cortexes[1,i] = cortex_cuda.create_cortex_from_fields(loc[i], rgb=False)

for ind,img in enumerate(img_sizes):
    retinas[0,4,ind] = retina_cuda.create_retina(loc50k, coeff50k, (img[1],img[0],3), (int(img[0]/2), int(img[1]/2)))
    retinas[1,4,ind] = retina_cuda.create_retina(loc50k, coeff50k, (img[1], img[0]), (int(img[0]/2), int(img[1]/2)))
cortexes[0][4] = cortex_cuda.create_cortex_from_fields_and_locs(L, R, L_loc, R_loc, cort_size, gauss100=G, rgb=True)
cortexes[1][4] = cortex_cuda.create_cortex_from_fields_and_locs(L, R, L_loc, R_loc, cort_size, gauss100=G, rgb=False)

#### TRACKBAR
def nothing(x):
    pass
cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Retina','Settings',0,4,nothing)
cv2.createTrackbar('Image size', 'Settings',0,5,nothing)
cv2.createTrackbar('Color mode','Settings',0,1,nothing)
###

camid = -1
cap = cv2.VideoCapture(camid)

while not cap.isOpened():
    print 'retrying\n'
    cv2.VideoCapture(camid).release()
    cap = cv2.VideoCapture(camid)
    camid += 1

r, img = cap.read()
while not r: r, img = cap.read()
print 'Press q to quit'
while True:
    i = cv2.getTrackbarPos('Retina', 'Settings')
    img_ind = cv2.getTrackbarPos('Image size', 'Settings')
    rgb = False if cv2.getTrackbarPos('Color mode', 'Settings') == 0 else True

    r, img = cap.read()
    img = cv2.resize(img, img_sizes[img_ind])
    if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if r:
	start = time.time()
        V_c = retinas[0 if rgb else 1][i%5][img_ind].sample(img) # sample with CUDA
        inv_c = retinas[0 if rgb else 1][i%5][img_ind].inverse(V_c) # inverse with CUDA
    
        l_c = cortexes[0 if rgb else 1][i%5].cort_image_left(V_c) # left cortical image CUDA
        r_c = cortexes[0 if rgb else 1][i%5].cort_image_right(V_c) # right cortical image CUDA
        end = time.time()
	print end-start

        c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1)

        cv2.namedWindow("inverse CUDA", cv2.WINDOW_NORMAL)
        cv2.imshow("inverse CUDA", inv_c)
        cv2.namedWindow("cortex CUDA", cv2.WINDOW_NORMAL)
        cv2.imshow("cortex CUDA", c_c)
        
    key = cv2.waitKey(10)
    if key == ord('q'): break
