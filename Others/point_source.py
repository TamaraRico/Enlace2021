# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:53:27 2021

@author: tamar
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Read Images
# img = mpimg.imread('tree.jpeg')
  
# # Output Images
# plt.imshow(img)

widht=128
height=128

hr_sx = np.linspace(width,height,10).reshape(-1,1)
hr_sy = np.linspace(-p.width,p.width,p.Q).reshape(-1,1)

r_k = np.random.choice(hr_sx[:,0], (p.k,2)) # assming square image
print(r_k.shape)

truth_img = np.zeros((p.Q,p.Q))
for x,y in r_k:
    indx,_ = find_nearest(hr_sx,x)
    indy,_ = find_nearest(hr_sy,y)
    truth_img[indx,indy] = 1.0

truth_img = truth_img.T
plt.imshow(truth_img)
plt.show()
lr_im = hr_im[::4, ::4]