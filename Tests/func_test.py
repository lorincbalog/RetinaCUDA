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
print coeff[0].shape
print type(coeff[0])
coeff[1] = scipy.io.loadmat(mat_data+'/coeff4k.mat')['M4k']
coeff[2] = scipy.io.loadmat(mat_data+'/coeff1k.mat')['M1k']
coeff[3] = scipy.io.loadmat(retina_path+'/coeffv2_1.mat')['coeffv2']
loc[0] = scipy.io.loadmat(mat_data+'/locations.mat')['ind']
print loc[0].shape
print type(loc[0])
loc[1] = scipy.io.loadmat(mat_data+'/loc4k.mat')['ind4k']
loc[2] = scipy.io.loadmat(mat_data+'/loc1k.mat')['ind1k']
loc[3] = scipy.io.loadmat(retina_path+'/locv2_1.mat')['locv2']
ret50k = scipy.io.loadmat(retina_path+'/ret50k_sorted.mat')
with open(retina_path + '/ret50k_loc.pkl', 'rb') as handle:
    loc50k = pickle.load(handle)
with open(retina_path + '/ret50k_coeff.pkl', 'rb') as handle:
    coeff50k = pickle.load(handle)

def correctness_test(loc, coeff, cap, rgb=False):
    '''
    CUDA code uses the minimal initialisation from the host,
    all tracatable values are computed on the GPU
    Get an image from the camera, generate inverse and cortical image 
    with both implementation and subtract the results
    '''
    r, img = cap.read()
    if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create CUDA objects to pass to evaluation
    ret = retina_cuda.create_retina(loc, coeff, img.shape, (int(img.shape[1]/2), int(img.shape[0]/2)), None)
    cort = cortex_cuda.create_cortex_from_fields(loc, rgb=rgb)

    while ord('q') != cv2.waitKey(10):
        r, img = cap.read()
        if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if r:
            '''
            Sample the image img with CUDA retina ret, inverse transform it with ret and 
            create the cortical image with CUDA cortex cort
            Sample and generate retina and cortical images from img with Piotrs's code
            Visually compare the results by showing the subtraction of the generatd images
            '''
            V_c = ret.sample(img) # sample with CUDA
            inv_c = ret.inverse(V_c) # inverse with CUDA
        
            l_c = cort.cort_image_left(V_c) # left cortical image CUDA
            r_c = cort.cort_image_right(V_c) # right cortical image CUDA
            c_c = np.concatenate((np.rot90(l_c),np.rot90(r_c,k=3)),axis=1) #concatenate the results into one image
        
            # create Piotr's retina and cortical images
            
            GI = retina.gauss_norm_img(int(img.shape[1]/2), int(img.shape[0]/2), coeff, loc, img.shape, rgb)
            L, R = cortex.LRsplit(loc)
            L_loc, R_loc = cortex.cort_map(L, R)
            L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)
            V_p = retina.sample(img, img.shape[1]/2, img.shape[0]/2, coeff, loc, rgb)
            inv_p = retina.inverse(V_p, img.shape[1]/2, img.shape[0]/2, coeff, loc, GI, img.shape, rgb)
            l_p, r_p = cortex.cort_img(V_p, L, L_loc, R, R_loc, cort_size, G)
            c_p = np.concatenate((np.rot90(l_p[:l_c.shape[0],:]),np.rot90(r_p[:r_c.shape[0],:],k=3)),axis=1)
            

            # show CUDA results
            cv2.namedWindow("inverse CUDA", cv2.WINDOW_NORMAL)
            cv2.imshow("inverse CUDA", inv_c)
            cv2.namedWindow("cortex CUDA", cv2.WINDOW_NORMAL)
            cv2.imshow("cortex CUDA", c_c)
            
            # show Piotr's results
            cv2.namedWindow("inverse Piotr", cv2.WINDOW_NORMAL)
            cv2.imshow("inverse Piotr", inv_p)
            cv2.namedWindow("cortex Piotr", cv2.WINDOW_NORMAL)
            cv2.imshow("cortex Piotr", c_p)
            
            # show the difference of the images
            cv2.namedWindow("inverse diff", cv2.WINDOW_NORMAL)
            cv2.imshow("inverse diff", np.power((inv_c - inv_p),2) * 255)
            cv2.namedWindow("cortex diff", cv2.WINDOW_NORMAL)
            cv2.imshow("cortex diff", np.power((c_c - c_p),2) * 255)

def compatibility_test(loc, coeff, cap, rgb=False):
    '''
    CUDA code uses different initialisations,
    passed parameters are the results of Piotr's code
    Initialise retina and cortex with external parameters
    Process camera stream
    '''
    r, img = cap.read()
    if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get parameters calculated by Piotr's code
    GI = retina.gauss_norm_img(int(img.shape[1]/2), int(img.shape[0]/2), coeff, loc, img.shape, rgb)
    L, R = cortex.LRsplit(loc)
    L_loc, R_loc = cortex.cort_map(L, R)
    L_loc, R_loc, G, cort_size = cortex.cort_prepare(L_loc, R_loc)

    # CUDA
    # first retina creates everything on the GPU, proved to be identical with Piotr's implementation
    ret0 = retina_cuda.create_retina(loc, coeff, img.shape, (img.shape[1]/2, img.shape[0]/2), None)
    # second retina uses the GI from Piotr
    ret1 = retina_cuda.create_retina(loc, coeff, img.shape, (img.shape[1]/2, img.shape[0]/2), GI)
    # first cortex creates everything on the GPU, proved to be identical with Piotr's implementation
    cort0 = cortex_cuda.create_cortex_from_fields(loc, rgb=rgb)
    # second cortex gets all the parameters from Piotr's code
    cort1 = cortex_cuda.create_cortex_from_fields_and_locs(L, R, L_loc, R_loc, (cort0.cort_image_size[0], cort_size[1]), gauss100=G, rgb=rgb)
    
    # read camera stream
    while ord('q') != cv2.waitKey(10):
        r, img = cap.read()
        if not rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if r:
            '''
            Sample the image img with ret0 and ret1, inverse transform the image vectors
            create the cortical image with cort0 and cort1
            Visually compare the results by showing the subtraction of the generatd images
            '''
            V0 = ret0.sample(img) # sample with reference ret
            inv0 = ret0.inverse(V0) # inverse with reference ret
        
            l_c0 = cort0.cort_image_left(V0) # left cortical image reference cort
            r_c0 = cort0.cort_image_right(V0) # right cortical image refernce cort
            c_c0 = np.concatenate((np.rot90(l_c0),np.rot90(r_c0,k=3)),axis=1) #concatenate the results into one image

            V1 = ret0.sample(img) # sample with reference ret
            inv1 = ret0.inverse(V0) # inverse with reference ret
        
            l_c1 = cort1.cort_image_left(V1) # left cortical image reference cort
            r_c1 = cort1.cort_image_right(V1) # right cortical image refernce cort
            c_c1 = np.concatenate((np.rot90(l_c1[:,:]),np.rot90(r_c1[:,:],k=3)),axis=1) #concatenate the results into one image

            # sampling error between the two instance
            print('Sampling difference: %f' %  np.sum(V1 - V0))
            # show CUDA results
            cv2.namedWindow("inverse ref", cv2.WINDOW_NORMAL)
            cv2.imshow("inverse ref", inv0)
            cv2.namedWindow("cortex ref", cv2.WINDOW_NORMAL)
            cv2.imshow("cortex ref", c_c0)
            
            # show Piotr's results
            cv2.namedWindow("inverse toprove", cv2.WINDOW_NORMAL)
            cv2.imshow("inverse toprove", inv1)
            cv2.namedWindow("cortex toprove", cv2.WINDOW_NORMAL)
            cv2.imshow("cortex toprove", c_c1)
            
            # show the difference of the images
            print len(np.nonzero(inv0-inv1)[0])/(img.shape[0]*img.shape[1])
            cv2.namedWindow("inverse diff", cv2.WINDOW_NORMAL)
            cv2.imshow("inverse diff", np.abs(inv0 - inv1) * 255)
            cv2.namedWindow("cortex diff", cv2.WINDOW_NORMAL)
            cv2.imshow("cortex diff", np.abs(c_c0 - c_c1) * 255)    

if __name__ == "__main__":
    camid = -1
    cap = cv2.VideoCapture(camid)

    while not cap.isOpened():
        print 'retrying\n'
        cv2.VideoCapture(camid).release()
        cap = cv2.VideoCapture(camid)
        camid += 1

    #correctness_test(loc[0], coeff[0], cap, rgb=True)
    compatibility_test(loc[0], coeff[0], cap, False)
    #compatibility_test(loc50k, coeff50k, cap, False)
    
    
    quit()