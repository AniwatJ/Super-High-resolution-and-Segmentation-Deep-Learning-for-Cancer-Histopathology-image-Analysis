# Super-High-resolution-and-Segmentation-Deep-Learning-for-Cancer-Histopathology-image-Analysis


## Introudction


 Pathology diagnosis is routine work performed by skill-full pathologists and experts. They prepare the stained specimen on a glass slide and observe it under a microscope. In order to accurately diagnose the sample, the digital camera is used to capture the entire slide with a scanner so-called the whole slide image (WSI), and it is saved as digital image. Therefore, computer vision plays a vital role in digital pathological image analysis.  However, there are several factors that a digital pathological image can be unresolve and has poor quality such as the low-quality digital camera, image compression for data storage, speed of motorize stage for the whole slide scanning, etc.  This could lead to misdiagnosis if pathologies keep diagnosing the low-resolution (LR) images. Moreover, Cell identification and quantification from H&E stained images can be time consuming and tedious for pathologies, since they have to consider every single cell and it could result in making mistakes for analysis. Therefore, the automatic nuclei segmentation software can help them to circumvent this problem and reduce workload of pathologists.   
In this paper, we propose a convolution neural network based on super resolution generative adversarial network (SRGAN). to enhance low resolution (LR) images. This approach can provide considerably resolution enhancement for poor quality images. To train SRGAN model, it requires data sets that consist of high-resolution images (ground truth) and their corresponding low-resolution images. We used a commercial microscope  to prepare dataset for training SRGAN. Peak Signal to Noise Ratio (PSNR) and Structural image similarity method (SSIM) of the results generated from our model are 25.389 dB and 0.899, respectively. This is promising results as it is close to the evaluation of SRGAN works  that they heavily did experiment using this architecture with the extremely huge data bases of general images. Furthermore, we also apply U-net model, which is the convolutional network architecture, for image segmentation and it is widely used for biomedical image segmentation. To train the U-net model, we need to H&E images and their binary mask for nuclei area. It is time consuming for the dataset  preparation, thus we used dataset from the cancer imaging archive (https://doi.org/10.7937/tcia.2019.4a4dkp9u) to train our U-net model.  The accuracy of the segmentation model is approximately 83%. 




![image](https://user-images.githubusercontent.com/83015448/115782089-4184c780-a389-11eb-85c6-5437c4d13272.png)
**Figure 1** Workflow of super high resolution and segmentation deep learning. (a) fresh tissue, (b) the corresponding H&E stained tissue slide, (c) commercial microscope for capturing the H&E stained tissue slide images, (d) high-resolution image acquired by the microscope, (e) simulated low resolution images, (f) SRGAN network, (g) unseen low resolution image, (h) Generator model from SRGAN, (i) generated high-resolution image, (j) U-net model for segmentation, (k) segmented H&E image


Figure1 shows the works flow of this work. First, we prepared the breast tumor H&E slides and used microscope to image these slides with high-magnification objective lens (40x), then we degraded the images by down sampling and adding noise. Therefore, we have corresponding ground truth (high resolution images) and low-resolution for training the Super-resolution generative adversarial network (SRGAN).  Eventually, the well-trained SRGAN model(figure1 b) is used to enhance unseen the low resolution images by generating another high resolution images. Furthermore, we take the advantages of these generated images to characterize the cell images as their resolution is substantially improved and they contain considerable detail. Particularly, some nuclei are close to each other, but they cannot resolve as the poor image quality, thus the software will segment them as a single cell. However, SRGAN can tackle this problem following by applying U-net model to the generated images (the output of SRGAN) in order to segment and quantify the cells so that we can characterize density, size, and morphology of the nuclei.  






# Method

2.1	Super resolution Generative adversarial network (SRGAN) 

The purpose for Super-resolution GAN (SRGAN) is to synthesize or generate super-resolution images from low-resolutions images, with more meticulous detail.  SRGAN consists of a generator model (Figure 2(a)) and discriminator model(Figure2(b)). The generator model takes a low-resolution image as the input and generates a high-resolution image after the convolution, residual, and upsampling layers. The discriminator is used to distinguish the generated image from the ground-truth image by taking them as the input and providing probability as the output.   The goal of SRGAN is to train the generator model to synthesize the image that can fool the discriminator model completely. To achieve this, we need to use the number of images as the data set to train to model as well as fine tune the hyper parameters thoroughly.   To train SRGAN, we started training the generator model first by freezing the generator model. Next step, we used adversarial network to train the generator model. The adversarial network (Figure 2(c)) is the combined models which are the generator model, discriminator model and VGG19. The VGG 19 is used as the features extractor.




 
 ![image](https://user-images.githubusercontent.com/83015448/115819319-9ac21a80-a3cc-11eb-9104-9ae5c22f232b.png)

 
 
 
 **Figure 2**   Super resolution generative adversarial network (SRGAN). (a) Generator model, (b) Discriminator model, (c) The combined models so-called adversarial model for training Generator model. 



2.2 The U-net architecture for image segmentation. 

The conventional CNNs for image segmentation task has two main components, which are encoder and decoder. Similarly, the U-net architecture also has these two parts, but the skip connection is the key component that allows U-net can surpass the conventional method and have better performance. Actually, this concept is the same as residual block that the input will concatenate to output at the same dimension. For the U-net case, the information from encoder layers (coarse information, left side) will concatenate to decoder layers (fine spatial information, right side). 


![image](https://user-images.githubusercontent.com/83015448/115818942-de685480-a3cb-11eb-9811-71268c547e71.png)

**Figure 5** U-net architecture for H&E images segmentation.  Every single blue box corresponds to a multi-channel feature map. The value over the boxes represents the number of channels. The value at the lower left edge of each box is the x-y size. The white boxes denote copied feature maps. The arrows indicate the different operations.
 
 
  
  
  
  
  
  
  
  
  
  
  
  
 
 
 

 















## Result 

The ultimate goal of SRGAN is to have the well-trained Generator model to reconstruct high resolution images. In terms of particle, we cannot feed the large image to generator model due to the computation restriction. Therefore, the large images are divided into serval small images. These images need to have the overlapping area in order to stitch them back to get the same field of view (FOV) as the original large image.   Figure 2 shows the result of SRGAN. First, three low resolution images (figure2 (a)) were enhanced resolution and quality by Generator model (figure2(b)). The figure 2 (c) shows generated high-resolution images which are the output of the generator model. The stitching software was applied to these generated images and provided the larger FOV (figure 2 (e)).  The figure 2 (f) illustrates the stitch images from nine generated images. Because of the impressive quality of generated images from our generator model, we take advantages of them to characterize the nuclei using segmentation. U-net model (figure 2(g)) is applied to the generated image and the result is shown in figure 2(h).



 ![image](https://user-images.githubusercontent.com/83015448/115810775-86762180-a3bc-11eb-9efd-8fbed44575b0.png)

 
 
 

## Usage ##

  * The training script for SRGAN is placed in /training/SRGAN/SRGAN_train.py
   
  * The training script for U-net is placed in /training/U-net/Unet_train.py
   
   * The final implementration script for Super resoltuion enhancement, segmenation, and stitiching is placed in /Implementation/Final_implementation.py
   (This implementation requires two models, which are generator.h5 and Unet.h5)









## Troubleshootings / Discussion
If you have any problem using This work, or you are interested in getting involved in developing it, feel free to email me. (aniwatbme1@gmail.com)









