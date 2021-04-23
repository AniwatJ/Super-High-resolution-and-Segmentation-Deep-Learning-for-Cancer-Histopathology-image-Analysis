
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 07:41:38 2021

@author: Aniwat Juhong
"""

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
import glob
import os 
import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
#from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import img_to_array, load_img, save_img

from keras.models import Model, load_model



from scipy.misc import imsave
from scipy.misc.pilutil import imread,imresize
from matplotlib import pyplot as plt
import cv2

from PIL import Image 

from skimage.io import imread, imshow, imread_collection, concatenate_images



###pip install opencv-contrib-python

##### Generator model ######
def residual_block(x):
    """
    Residual blok
    """
    filters=[64, 64]
    kernel_size=3
    strides=1
    padding ="same"
    momentum=0.8
    activation="relu"
    res=Conv2D(filters=filters[0],kernel_size=kernel_size,strides=strides,padding=padding)(x)
    res=Activation(activation=activation)(res)
    res=BatchNormalization(momentum=momentum)(res)
    
    res=Add()([res,x])
    return res



def build_generator():
    ##### Define hyperparameters
    residual_blocks=16
    momentum=0.8
    input_shape=(64,64,3)

    input_layer=Input(shape=input_shape)
    gen1=Conv2D(filters=64,kernel_size=9,padding='same',activation='relu')(input_layer)
    
    res=residual_block(gen1)
    for i in range(residual_blocks-1):
        res=residual_block(res)
    
    gen2=Conv2D(filters=64, kernel_size=3, strides=1,padding='same')(res)
    gen2=BatchNormalization(momentum=momentum)(gen2)


    #take the sum of the output from the pre-residual block(gen1) and tehe post-residual block (gen2)
    gen3=Add()([gen2,gen1])
    
    # umspaling block
    gen4=UpSampling2D(size=2)(gen3)
    gen4=Conv2D(filters=256, kernel_size=3,strides=1,padding='same')(gen4)
    gen4=Activation('relu')(gen4)
    
    
    gen5=UpSampling2D(size=2)(gen4)
    gen5=Conv2D(filters=256, kernel_size=3,strides=1,padding='same')(gen5)
    gen5=Activation('relu')(gen5)
    
   
    #Output Convolution layer
    gen6=Conv2D(filters=3,kernel_size=9,strides=1,padding='same')(gen5)
    output=Activation('tanh')(gen6)
    
    
    model=Model(inputs=[input_layer],outputs=[output],name='generator')
    
    return model

    



def ST(Right_image, Left_image):
    
    img=Left_image
    img_=Right_image
    
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    

    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
            matches = np.asarray(good)

    
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 2)
        #R =cv2.warpImages(img, img_, H)

    else:
        raise AssertionError("Can’t find enough keypoints.")


    dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img


    DST=dst
    
    Y_C=dst.shape[1]-200
    DST=dst[0:400,0:Y_C]

    return DST






def ST2(Right_image, Left_image):
    
    img=Left_image
    img_=Right_image
    
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    

    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
            matches = np.asarray(good)

    
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        #R =cv2.warpImages(img, img_, H)

    else:
        raise AssertionError("Can’t find enough keypoints.")


    dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img


    DST=dst
    
    Y_C=dst.shape[1]-128
    DST=dst[0:256,0:Y_C]

    return DST



def ST3(Right_image, Left_image):
    
    img=Left_image
    img_=Right_image
    
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    

    good = []
    for m in matches:
        if m[0].distance < 0.6*m[1].distance:
            good.append(m)
            matches = np.asarray(good)

    
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 1)
        #R =cv2.warpImages(img, img_, H)

    else:
        raise AssertionError("Can’t find enough keypoints.")


    dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img


    DST=dst
    Y_C=dst.shape[1]-128
    DST=dst[0:512,0:Y_C]

    return DST



def ST4(Right_image, Left_image):
    
    img=Left_image
    img_=Right_image
    
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    

    good = []
    for m in matches:
        if m[0].distance < 0.6*m[1].distance:
            good.append(m)
            matches = np.asarray(good)

    
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 1)
        #R =cv2.warpImages(img, img_, H)

    else:
        raise AssertionError("Can’t find enough keypoints.")


    dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img


    DST=dst
    Y_C=dst.shape[1]-128
    DST=dst[0:512,0:Y_C]

   
    return DST






low_resolution_shape=(64,64,3)
High_resolution_shape=(256,256,3)

generator=build_generator()
generator.load_weights('generator6000.h5')
#generator.load_weights('generator_finalV2.h5')


################################### Apply SRGAN #############################################################





y1= 22
for i in range(1,3):
    
    if i==1:
        print(i+5)
        low_resolution_images = []
        pathHE11 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,i+5)
        HE11 =imread(pathHE11) 
        img=HE11;
        # plt.figure(22)
        # plt.imshow(HE11)
        imshow(img)
        plt.show()
        img = img.astype(np.float32)
        low_resolution_images.append(img)
        low_resolution_images = np.array(low_resolution_images)
        print("%d %d"%(y1,i+5))
     
        
        low_resolution_images = low_resolution_images / 127.5 - 1
        generated_images = generator.predict_on_batch(low_resolution_images)
        Generated_IMG=generated_images+1
        Generated_IMG=Generated_IMG*127.5
        Generated_IMG = Generated_IMG.astype(np.uint8)
        GM1=Generated_IMG[0,:,:,:]
        imshow(GM1)
        plt.show()
        
        
        
        
        
        
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate1.jpg", GM1)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution1.jpg", HE11)
        
        low_resolution_images2 = []
        pathHE22 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,i+6)
        HE22 =imread(pathHE22) 
        img2=HE22;
        imshow(img2)
        plt.show()
        img2 = img2.astype(np.float32)
        low_resolution_images2.append(img2)
        low_resolution_images2 = np.array(low_resolution_images2)
        low_resolution_images2 = low_resolution_images2 / 127.5 - 1
        
        generated_images2 = generator.predict_on_batch(low_resolution_images2)
        Generated_IMG2=generated_images2+1
        Generated_IMG2=Generated_IMG2*127.5
        Generated_IMG2 = Generated_IMG2.astype(np.uint8)
        GM2=Generated_IMG2[0,:,:,:]
        imshow(GM2)
        plt.show()
        
        
        
        
        
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate2.jpg", GM2)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution2.jpg", HE22)
    
        dst1=ST2(Right_image=GM2,Left_image=GM1)
        imshow(dst1)
        plt.show()
        #plt.imshow(dst1)
        c=i+6
        
    else:
        c=c+1
        print(c)
        low_resolution_images3 = []
        pathHE33 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,c)
        HE33 =imread(pathHE33) 
        img3=HE33;
        imshow(img3)
        plt.show()
        img3 = img3.astype(np.float32)
        low_resolution_images3.append(img3)
        low_resolution_images3 = np.array(low_resolution_images3)
        low_resolution_images3 = low_resolution_images3 / 127.5 - 1
        
        generated_images3 = generator.predict_on_batch(low_resolution_images3)
        Generated_IMG3=generated_images3+1
        Generated_IMG3=Generated_IMG3*127.5
        Generated_IMG3 = Generated_IMG3.astype(np.uint8)
        GM3=Generated_IMG3[0,:,:,:]
        imshow(GM3)
        plt.show()
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate3.jpg", GM3)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution3.jpg", HE33)
        dst2=ST2(Right_image=GM3,Left_image=dst1)
        
        
        imshow(dst2)
        plt.show()






y1= 23
for i in range(1,3):
    print(i)
    if i==1:
        print(i+5)
        low_resolution_images = []
        pathHE11 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,i+5)
        HE11 =imread(pathHE11) 
        img=HE11;
           
        imshow(img)
        plt.show()
        img = img.astype(np.float32)
        low_resolution_images.append(img)
        low_resolution_images = np.array(low_resolution_images)
        low_resolution_images = low_resolution_images / 127.5 - 1
        
        generated_images = generator.predict_on_batch(low_resolution_images)
        Generated_IMG=generated_images+1
        Generated_IMG=Generated_IMG*127.5
        Generated_IMG = Generated_IMG.astype(np.uint8)
        GM1=Generated_IMG[0,:,:,:]
        imshow(GM1)
        plt.show()
        
        


        low_resolution_images2 = []
        pathHE22 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,i+6)
        HE22 =imread(pathHE22) 
        img2=HE22;
        imshow(img2)
        plt.show()
        img2 = img2.astype(np.float32)
        low_resolution_images2.append(img2)
        low_resolution_images2 = np.array(low_resolution_images2)
        low_resolution_images2 = low_resolution_images2 / 127.5 - 1
        
        generated_images2 = generator.predict_on_batch(low_resolution_images2)
        Generated_IMG2=generated_images2+1
        Generated_IMG2=Generated_IMG2*127.5
        Generated_IMG2 = Generated_IMG2.astype(np.uint8)
        GM2=Generated_IMG2[0,:,:,:]
        imshow(GM2)
        plt.show()
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate5.jpg", GM2)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution5.jpg", HE22)
    
        dst11=ST2(Right_image=GM2,Left_image=GM1)
        imshow(dst11)
        plt.show()
        
        c=i+6
        
    else:
        c=c+1
        print(c)
        low_resolution_images3 = []
        pathHE33 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,c)
        HE33 =imread(pathHE33) 
        img3=HE33;
        imshow(img3)
        plt.show()
        img3 = img3.astype(np.float32)
        low_resolution_images3.append(img3)
        low_resolution_images3 = np.array(low_resolution_images3)
        low_resolution_images3 = low_resolution_images3 / 127.5 - 1
        
        
        
        generated_images3 = generator.predict_on_batch(low_resolution_images3)
        Generated_IMG3=generated_images3+1
        Generated_IMG3=Generated_IMG3*127.5
        Generated_IMG3 = Generated_IMG3.astype(np.uint8)
        GM3=Generated_IMG3[0,:,:,:]   
        imshow(GM3)
        plt.show()

        dst22=ST2(Right_image=GM3,Left_image=dst11)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate6.jpg", GM3)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution6.jpg", HE33)
        
        imshow(dst22)
        plt.show()








y1= 24
for i in range(1,3):
    print(i)
    if i==1:
        print(i+5)
        low_resolution_images = []
        pathHE11 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,i+5)
        HE11 =imread(pathHE11) 
        img=HE11;
        imshow(img)
        plt.show()

        img = img.astype(np.float32)
        low_resolution_images.append(img)
        low_resolution_images = np.array(low_resolution_images)
        low_resolution_images = low_resolution_images / 127.5 - 1
        
        generated_images = generator.predict_on_batch(low_resolution_images)
        Generated_IMG=generated_images+1
        Generated_IMG=Generated_IMG*127.5
        Generated_IMG = Generated_IMG.astype(np.uint8)
        GM1=Generated_IMG[0,:,:,:]
        imshow(GM1)
        plt.show()

        
        
        
        
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate7.jpg", GM1)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution7.jpg", HE11)

        low_resolution_images2 = []
        pathHE22 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,i+6)
        HE22 =imread(pathHE22) 
        img2=HE22;
        imshow(img2)
        plt.show()

        img2 = img2.astype(np.float32)
        low_resolution_images2.append(img2)
        low_resolution_images2 = np.array(low_resolution_images2)
        low_resolution_images2 = low_resolution_images2 / 127.5 - 1
        
        generated_images2 = generator.predict_on_batch(low_resolution_images2)
        Generated_IMG2=generated_images2+1
        Generated_IMG2=Generated_IMG2*127.5
        Generated_IMG2 = Generated_IMG2.astype(np.uint8)

        GM2=Generated_IMG2[0,:,:,:]
        imshow(GM2)
        plt.show()
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate8.jpg", GM2)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution8.jpg", HE22)
    
        dst1=ST2(Right_image=GM2,Left_image=GM1)
        imshow(dst1)
        plt.show()
        c=i+6
        
    else:
        c=c+1
        print(c)
        low_resolution_images3 = []
        pathHE33 = 'C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\LOW_Resolution4\%dImage%d.png'%(y1,c)
        HE33 =imread(pathHE33) 
        img3=HE33;
        imshow(img3)
        plt.show()
        img3 = img3.astype(np.float32)
        low_resolution_images3.append(img3)
        low_resolution_images3 = np.array(low_resolution_images3)
        low_resolution_images3 = low_resolution_images3 / 127.5 - 1
        
        generated_images3 = generator.predict_on_batch(low_resolution_images3)
        Generated_IMG3=generated_images3+1
        Generated_IMG3=Generated_IMG3*127.5
        Generated_IMG3 = Generated_IMG3.astype(np.uint8)

        GM3=Generated_IMG3[0,:,:,:]
        imshow(GM3)
        plt.show()
        
        
        
        
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Generate9.jpg", GM3)
        # plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\Low_resolution9.jpg", HE33)
        
        
        
        
        dst222=ST2(Right_image=GM3,Left_image=dst1)
        imshow(dst222)
        plt.show()
        







R_dst2=cv2.rotate(dst2,cv2.ROTATE_90_COUNTERCLOCKWISE)
R_dst22=cv2.rotate(dst22,cv2.ROTATE_90_COUNTERCLOCKWISE)
FOV_DST=ST3(Right_image=R_dst22,Left_image=R_dst2)




R_dst222=cv2.rotate(dst222,cv2.ROTATE_90_COUNTERCLOCKWISE)
FOV_DST2=ST4(Right_image=R_dst222,Left_image=FOV_DST)


# plt.imsave("C:\Research\Deep_learning_project\GAN\H&E_super_resolution_image\Results\GIMG.jpg", FOV_DST2)


plt.figure(2)
plt.imshow(FOV_DST2)

Final=FOV_DST2[0:512,0:512]

plt.figure(3)
plt.imshow(Final)
# plt.axis('off')
# plt.savefig('test.png',dpi=80)


############################## Implement segementation #######################


model = load_model('Unet.h5')
model.summary()

img_size = (256, 256)
X_input = np.zeros((1, img_size[0], img_size[1], 3), dtype = np.uint8)

dsize = (256, 256)
img = Final[:, : , : 3]

img = cv2.resize(img, dsize)


imshow(img)
plt.show()
X_input[0] = img
preds_test = model.predict(X_input, verbose = 1)


preds_test_t = (preds_test > 0.95).astype(np.uint8)

Result = preds_test_t[0]


imshow(np.squeeze(preds_test_t[0]))
plt.show()




