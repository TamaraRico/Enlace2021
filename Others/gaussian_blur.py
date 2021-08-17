# -*- coding: utf-8 -*-

#from wand.image import Image
from PIL import Image, ImageFilter 
import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image
from skimage.util import random_noise

#im = Image.open("test.jpg")
# convert PIL Image to ndarray
#im_arr = np.asarray(im)

# random_noise() method will convert image in [0, 255] to [0, 1.0],
# inherently it use np.random.normal() to create normal distribution
# and adds the generated noised back to image
# noise_img = random_noise(im_arr, mode='gaussian', var=0.05**2)
# noise_img = (255*noise_img).astype(np.uint8)

# img = Image.fromarray(noise_img)
#img.show()
# creating a image object 
im1 = Image.open(r"C:\Users\tamar\OneDrive\Escritorio\Tamara uabc\4to semestre\Enlace 2021\Códigos\output.jpg") 
  
# applying the Gaussian Blur filter 
im2 = im1.filter(ImageFilter.GaussianBlur(radius = 5))
#im2.show()

# transforms the image to a numpy array
im_arr = np.asarray(im2)

# changes the scale of the image /  
lr_im = im_arr[::4, ::4] 

im2=Image.fromarray(lr_im)

# Resize smoothly down to 16x16 pixels
imgSmall = im2.resize((60,60),resample=Image.BILINEAR)

# Scale back up using NEAREST to original size
result = imgSmall.resize(im2.size,Image.NEAREST)

# im3=Image.fromarray(lr_im)
# im3.show() 

#lr_im = np.asarray(im3)

#im_arr = np.asarray(im2)
lr_im = np.asarray(result)

# Adds noise to the image using a function of the library
noise_img = random_noise(lr_im, mode='gaussian', var=0.01**2) #var=0.05**2)
noise_img = (255*noise_img).astype(np.uint8)

# transform the image back to image 
img = Image.fromarray(noise_img)

img.show()
img.save('trans.png')

#pix = np.array(im2)

#print(pix)

#lr_im = pix[::4, ::4]  

# Read image using Image() function
# with Image(filename ="outpput.jpeg") as img:
  
#     # Generate noise image using spread() function
#     img.noise("laplacian", attenuate = 1.0)
#     img.save(filename ="noisekoala2.jpeg")

## Construct additive noise (clip if less than 0)
# n = p.noise*np.random.randn(p.M**2,p.T)

# lr_im =np.random.randn(32,300)

# row,col,ch= pix.shape
# mean = 0
# var = 0.1
# sigma = var**0.5
# gauss = np.random.normal(mean,sigma,(row,col,ch))
# gauss = gauss.reshape(row,col,ch)
# noisy = pix + gauss

# #im2.show() 

# c = plt.imshow(noisy, cmap ='Greens',
#                     interpolation ='nearest', origin ='lower')
# plt.show()
  
# Read image using Image() function
# with Image(filename ="koala.jpeg") as img:
  
#     # Generate noise image using spread() function
#     img.noise("laplacian", attenuate = 1.0)
#     img.save(filename ="noisekoala2.jpeg")