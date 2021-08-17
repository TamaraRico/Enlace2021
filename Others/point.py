# -*- coding: utf-8 -*-

# import cv2
# import numpy as np

# width = 100
# height = 100

# # Make empty black image of size (100,100)
# img = np.zeros((height, width, 3), np.uint8)

# red = [0,0,255]

# # Change pixel (50,50) to red
# img[50,50] = red

# cv2.imshow('img', img)
# cv2.waitKey(0)

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#import random
import numpy as np

pixels=128
sources=10

# x-axis values
#x = [1,2,3,4,5,6,7,8,9,10]
x = np.random.randint(0,pixels,sources)
# y-axis values
#y = [2,4,5,7,6,8,9,11,12,12]
y = np.random.randint(0,pixels,sources)
  
# plotting points as a scatter plot
#plt.subplot(1,2,1)
plt.scatter(x, y, color= "green", 
            marker= ".", s=30)
  
# x-axis label
plt.xlabel('x - axis')
# frequency label
plt.ylabel('y - axis')
# plot title
plt.title('Point-source data')
# showing legend
plt.legend()

#plt.subplot(1,2,2)
#plt.scatter(plo, color= "green", marker= "*", s=128)
# function to show the plot
#plt.show()
#plt.savefig('point_sd.png')
#plt.savefig('foo.png', bbox_inches='tight')
# Saving the figure.
plt.savefig("output.jpg")
  
# Saving figure by changing parameter values
plt.savefig("output1", facecolor='b', bbox_inches="tight",
            pad_inches=0.3, transparent=True)