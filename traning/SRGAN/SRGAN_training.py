# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 07:41:38 2021

@author: Aniwat Juhong
"""


import glob
import os 
import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import img_to_array, load_img, save_img
from scipy.misc import imsave
from scipy.misc.pilutil import imread,imresize
from matplotlib import pyplot as plt


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


def build_discriminator():
    
    leakyrelu_alpha=0.2
    momentum=0.8
    input_shape=(256,256,3)
    
    input_layer=Input(shape=input_shape)
    
    dis1=Conv2D(filters=64,kernel_size=3,strides=2, padding='same')(input_layer)
    dis1=LeakyReLU(alpha=leakyrelu_alpha)(dis1)
    
    
    
    dis2=Conv2D(filters=64,kernel_size=3,strides=2, padding='same')(dis1)
    dis2=LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2=BatchNormalization(momentum=momentum)(dis2)
    
    
        
    dis3=Conv2D(filters=128,kernel_size=3,strides=2, padding='same')(dis2)
    dis3=LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3=BatchNormalization(momentum=momentum)(dis3)
    
    
    dis4=Conv2D(filters=128,kernel_size=3,strides=2, padding='same')(dis3)
    dis4=LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4=BatchNormalization(momentum=momentum)(dis4)
    
        
    dis5=Conv2D(filters=256,kernel_size=3,strides=2, padding='same')(dis4)
    dis5=LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5=BatchNormalization(momentum=momentum)(dis5)
    
    dis6=Conv2D(filters=256,kernel_size=3,strides=2, padding='same')(dis5)
    dis6=LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6=BatchNormalization(momentum=momentum)(dis6)
    
    dis7=Conv2D(filters=512,kernel_size=3,strides=2, padding='same')(dis6)
    dis7=LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7=BatchNormalization(momentum=momentum)(dis7)
    
    
    dis8=Conv2D(filters=512,kernel_size=3,strides=2, padding='same')(dis7)
    dis8=LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8=BatchNormalization(momentum=momentum)(dis8)
    
    
    dis9=Dense(units=1024)(dis8)
    dis9=LeakyReLU(alpha=0.2)(dis9)
    
    output=Dense(units=1,activation='sigmoid')(dis9)
    
    
    model=Model(inputs=[input_layer],outputs=[output],name='discriminator')
    return model



    
def build_vgg():
    
    input_shape=(256,256,3)
    vgg=VGG19(weights="imagenet")
    vgg.outputs=[vgg.layers[9].output]    
    input_layer=Input(shape=input_shape)
    features=vgg(input_layer)
    model=Model(inputs=[input_layer],outputs=[features])
    
    return model
    
    

    
    
def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    for img in images_batch:
        # Get an ndarray of the current image
        img1 = imread(img, mode='RGB')
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = imresize(img1, high_resolution_shape)
        img1_low_resolution = imresize(img1, low_resolution_shape)

        # Do a random horizontal flip
        if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)
            # pyplot.imshow(img1_high_resolution)
            # pyplot.axis('off')
            # pyplot.show()

            # pyplot.imshow(img1_low_resolution)
            # pyplot.axis('off')
            # pyplot.show()

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)

    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)



def save_images(low_resolution_image, original_image, generated_image, path):
    """
    Save low-resolution, high-resolution(original) and
    generated high-resolution images in a single image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(low_resolution_image)
    ax.axis("off")
    ax.set_title("Low-resolution")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(original_image)
    ax.axis("off")
    ax.set_title("Original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image)
    ax.axis("off")
    ax.set_title("Generated")

    plt.savefig(path)



    
### Training the SRGAN ###


########### Define hyperparameters ##############

data_dir="Dataset/*.*"


epochs=20000
batch_size=1
# Shape of low-resolution and hight-resoution images
low_resolution_shape=(64,64,3)
high_resolution_shape=(256,256,3)
common_optimizer=Adam(0.0002,0.5)



######################################### Compiles the newtowrks ###############################################

# # complie VGG19 network, don't need to traing VGG19 network
vgg=build_vgg()
vgg.trainable=False
vgg.compile(loss='mse',optimizer=common_optimizer,metrics=['accuracy'])    
    



discriminator=build_discriminator()
discriminator.compile(loss='mse',optimizer=common_optimizer,metrics=['accuracy'])

print("Compile Discriminator")


generator=build_generator()



# # ############ create an adversarial model >>>> GAN LOSS #################


input_high_resolution=Input(shape=high_resolution_shape)
input_low_resolution=Input(shape=low_resolution_shape)
generated_high_resolution_images=generator(input_low_resolution)



#Use VGG19 to extract featrue maps for the genrated images:
features=vgg(generated_high_resolution_images)
############# make the discriminator non-trainable as we don't want to train discriminator duing the training of the adversarial model
discriminator.trainable=False
# This probs represents the probabilities of the generated hight-resolution fake images
probs=discriminator(generated_high_resolution_images)




# create & complie adversarial network
adversarial_model=Model([input_low_resolution],[probs,features])
adversarial_model.compile(loss=['binary_crossentropy','mse'],loss_weights=[1e-3,1], optimizer=common_optimizer)






print("Compile Adversarial")



for epoch in range(epochs):
    print("Epoach:{}".format(epoch))
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                          low_resolution_shape=low_resolution_shape,
                                                                          high_resolution_shape=high_resolution_shape)

  # Normalize images >>>  range -1 to 1
    high_resolution_images = high_resolution_images / 127.5 - 1.
    low_resolution_images = low_resolution_images / 127.5 - 1.





 
# # Generate high-resolution images from low-resolution images
    generated_high_resolution_images = generator.predict(low_resolution_images)

    
    
#ValueError: Error when checking target: expected dense_28 to have shape (1, 1, 1) but got array with shape (16, 16, 1)    
    #real_labels = np.ones((batch_size, 16, 16, 1))
    
    
    real_labels = np.ones((batch_size,1,1,1))
    fake_labels = np.zeros((batch_size,1,1,1))



    d_loss_real = discriminator.train_on_batch(high_resolution_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(generated_high_resolution_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


    print("d_loss:",d_loss)
    
    high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                          low_resolution_shape=low_resolution_shape,
                                                                          high_resolution_shape=high_resolution_shape)
            # Normalize images
    high_resolution_images = high_resolution_images / 127.5 - 1.
    low_resolution_images = low_resolution_images / 127.5 - 1.

            # Extract feature maps for real high-resolution images
    image_features = vgg.predict(high_resolution_images)

            # Train the generator network


    g_loss = adversarial_model.train_on_batch([low_resolution_images],
                                              [real_labels, image_features])
    print("g_loss:", g_loss)


    #         # Sample and save images after every 100 epochs
    if epoch % 100 == 0:
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=batch_size,
                                                                              low_resolution_shape=low_resolution_shape,
                                                                              high_resolution_shape=high_resolution_shape)
                # Normalize images
        high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        generated_images = generator.predict_on_batch(low_resolution_images)

        for index, img in enumerate(generated_images):
              save_images(low_resolution_images[index], high_resolution_images[index], img,
              path="results/img_{}_{}".format(epoch, index))

# Save models
generator.save_weights("generator.h5")
discriminator.save_weights("discriminator.h5")



