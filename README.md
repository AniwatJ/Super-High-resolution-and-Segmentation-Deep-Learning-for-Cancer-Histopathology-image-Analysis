# Super-High-resolution-and-Segmentation-Deep-Learning-for-Cancer-Histopathology-image-Analysis


**Introduction.
**

  Pathology diagnosis is routine work performed by skill-full pathologists and experts. They prepare the stained specimen on a glass slide and observe it under a microscope. In order to accurately diagnose the sample, the digital camera is used to capture the entire slide with a scanner so-called whole slide image (WSI), and it is saved as digital image. Therefore, computer vision plays a vital role in digital pathological image analysis.  However, there are several factors that a digital pathological image can be unresolve and has poor quality such as the low-quality digital camera, image compression for data storage, speed of motorize stage for whole slide scanning, etc.  This could lead to misdiagnosis if pathologies keep diagnosing the low-resolution (LR) images. Moreover, Cell identification and quantification from H&E stained images can be time consuming and tedious for pathologies, since they have to consider every single cell and it could result in making mistakes for analysis. Therefore, the automatic nuclei segmentation software can help them to circumvent this problem and reduce workload of pathologists.   
In this work, we propose a convolution neural network based on super resolution generative adversarial network (SRGAN) [reference] to enhance low resolution (LR) images. This approach can provide considerably resolution enhancement for poor quality images. To train SRGAN model, it requires data sets that consist of high-resolution images (ground truth) and their corresponding low-resolution images. We used a commercial microscope (Nikon -xxx ) to prepare dataset SRGAN training. Peak Signal to Noise Ratio (PSNR) and Structural image similarity method (SSIM) of our model are 25.389 dB and 0.899, respectively. This is promising results as it is close to the evaluation of SRGAN works [ reference] that they heavily did experiment using this architecture with the extremely huge data bases of general images. Furthermore, we also apply U-net model, which is the convolutional network architecture, for image segmentation and it is widely used for biomedical image segmentation[reference]. To train the U-net model, we need to H&E images and their binary mask for nuclei area. It is time consuming for the preparation of this dataset, thus we used dataset from the cancer imaging archive (https://doi.org/10.7937/tcia.2019.4a4dkp9u) to train our U-net model.  The accuracy of the segmentation model is approximately 83%. 



![image](https://user-images.githubusercontent.com/83015448/115782089-4184c780-a389-11eb-85c6-5437c4d13272.png)



