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

#GIc = retina.gauss_norm_img(int(img.shape[1]/2), int(img.shape[0]/2), coeff50k, loc50k, img.shape, True)
GIg = retina.gauss_norm_img(int(img.shape[1]/2), int(img.shape[0]/2), coeff50k, loc50k, img.shape, False)
L, R = cortex.LRsplit(loc50k)
L_loc, R_loc = cortex.cort_map(L, R)
L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)

for i in range(0,4):
    retinas[0][i] = retina_cuda.create_retina(loc[i], coeff[i], img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
    retinas[1][i] = retina_cuda.create_retina(loc[i], coeff[i], (img.shape[0], img.shape[1]), (int(img.shape[1]/2), int(img.shape[0]/2)))
    cortexes[0][i] = cortex_cuda.create_cortex_from_fields(loc[i], rgb=True)
    cortexes[1][i] = cortex_cuda.create_cortex_from_fields(loc[i], rgb=False)

retinas[0][4] = retina_cuda.create_retina(loc50k, coeff50k, img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)))
retinas[1][4] = retina_cuda.create_retina(loc50k, coeff50k, (img.shape[0], img.shape[1]), (int(img.shape[1]/2), int(img.shape[0]/2)))
cortexes[0][4] = cortex_cuda.create_cortex_from_fields_and_locs(L, R, L_loc, R_loc, (cort0.cort_image_size[0], cort_size[1]), gauss100=GIg, rgb=True)
cortexes[1][4] = cortex_cuda.create_cortex_from_fields_and_locs(L, R, L_loc, R_loc, (cort0.cort_image_size[0], cort_size[1]), gauss100=GIg, rgb=False)

i = 1
rgb = False
print 'Press i to increase the retina size'
print 'Press q to decrease the retina size'
print 'Press c to switch between color and grayscale'
print 'Press q to quit'
while True:
    r, img = cap.read()
    if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if r:
        V_c = retinas[0 if rgb else 1][i%5].sample(img) # sample with CUDA
        inv_c = retinas[0 if rgb else 1][i%5].inverse(V_c) # inverse with CUDA
    
        l_c = cortexes[0 if rgb else 1][i%5].cort_image_left(V_c) # left cortical image CUDA
        r_c = cortexes[0 if rgb else 1][i%5].cort_image_right(V_c) # right cortical image CUDA
        c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1) #concatenate the results into one image

        # show CUDA results
        cv2.namedWindow("inverse CUDA", cv2.WINDOW_NORMAL)
        cv2.imshow("inverse CUDA", inv_c)
        cv2.namedWindow("cortex CUDA", cv2.WINDOW_NORMAL)
        cv2.imshow("cortex CUDA", c_c)
        
    key = cv2.waitKey(10)
    if key == ord('q'): break
    if key == ord('i'): 
        i += 1
        print 'New retina size: %i' % retinas[0 if rgb else 1][i%5].retina_size
    if key == ord('d'): 
        i -= 1
        print 'New retina size: %i' % retinas[0 if rgb else 1][i%5].retina_size
    if key == ord('c'): 
        rgb = not rgb
        print 'RGB mode' if rgb else 'Grayscale mode' 
