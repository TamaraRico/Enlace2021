# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter
from tqdm import trange
from matplotlib.pyplot import cm
#
# If you need any of the above packages, type: conda install <package-name> in your anaconda prompt.
#

def genGroundTruth(size, source_num):
    gt = np.zeros(size, dtype=np.float32) 

    # TO-DO: Write the remaining python code to randomly place all the sources
    # onto the gt image. Each pixel should have an equal change to be chosen
    # as one of the source points. NOTE that gt is a grayscale image with
    # decimal values between 0 and 1.

    #
    # [Write your implementation here]
    #
    brigth=255
    
    # for loop anidado

    for i in range(source_num):
        x=random.randrange(0, size[0])
        y=random.randrange(0, size[1])
        gt[x][y]=brigth

        
    return gt

def genPointSrcImage(size=(128, 128), kernel_std=3, noise_pwr=1, downsample_factor=3):
    source_num = random. randint(0,128) 
    gt = genGroundTruth(size, source_num)
    lr = gaussian_filter(gt, kernel_std)[::downsample_factor, ::downsample_factor] 

    # TODO: Add Gaussian noise (with the given noise power) to the
    # low-resolution image before returning

    #
    # [Write your implementation here]
    #
    
    #noise=0.1*noise_pwr*np.random.randn(32, 32)
    # noise=np.random.normal(0,0,lr.shape)
    noise = np.random.normal(0, 0, lr.shape)
    #blurry_image[noise]
    # for r in range(int(128/4.0)):
    #     for c in range(int(128/4)):
    #         for i in range(3):
    #             lr[r][c]+=noise[r][c]
    lr=lr+(0.01)*noise
    lr[lr < 0] = 0

    return (gt, lr)

'''
    Point-Source Data Generating Program
'''
if __name__ == "__main__":
    num_images = 100

    # create folder to store the images
    if not os.path.exists("data"):
        os.makedirs("data")

    # create dataset of ground truth images
    for i in trange(num_images):

        # create the ground truth and low-resolution images and save them
        gt, lr = genPointSrcImage()
        # plt.imsave(safe_sample_x, gt, cmap=cm.gray)
        plt.imsave(f"data/x_{i}.bmp", gt, cmap=cm.gray)
        plt.imsave(f"data/y_{i}.bmp", lr, cmap=cm.gray)

        # uncomment if you want to plot each image (be sure to comment out if
        # making a large number of points)
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(gt, cmap=cm.gray)
        # plt.subplot(122)
        # plt.imshow(lr/np.max(lr), cmap=cm.gray)
        # plt.show()
