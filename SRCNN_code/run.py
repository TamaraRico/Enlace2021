# -*- coding: utf-8 -*-

# for x in {1..10}; do (python script.py > /tmp/$x.log ) & done 

import os

if __name__ == "__main__":
    for scale in (2, 3, 4):
        for i in range(1000):
            cmd = f'python test.py --weights-file "weights/srcnn_x{scale}.pth" \
                           --image-file "data/Cars/data/y_{i}.png" \
                           --scale {scale}'
            os.system(cmd)


# import argparse
# import subprocess

# cmd='python test.py'
# numImages=3

# #def get_program_running():
# for i in range(numImages):
#     p=subprocess.Popen(cmd, shell= True)
#     subprocess.getoutput('--weights-file "srcnn_x3.pth" \ weigth="srcnn_x3.pth"')
#     # weigth="srcnn_x3.pth"
#     # img= "data/Cars/data/x_{i}.png"
#     # scale+=1
#     out, err=p.communicate()
#     # def get_i():
#     #     return i
#     print(err)
#     print(out)
    # if(scale==4):
    #     scale=2
        #return weigth, img, scale

    
# python test.py --weights-file "srcnn_x3.pth" \ weigth="srcnn_x3.pth"
# --image-file "data/butterfly_GT.bmp" \   "data/x_{i}.png"
# --scale x, x=2, x++, if x==4, x=2


# if __name__ == '__main__':
#     num_images = 1000

#     # create folder to store the images
#     if not os.path.exists("data"):
#         os.makedirs("data")

#     # create dataset of ground truth images
#     for i in trange(num_images):

#         # create the ground truth and low-resolution images and save them
#         gt, lr = genPointSrcImage()
#         plt.imsave(f"data/x_{i}.png", gt)
#         plt.imsave(f"data/y_{i}.png", lr)
        
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--weights-file', type=str, required=True)
#     parser.add_argument('--image-file', type=str, required=True)
#     parser.add_argument('--scale', type=int, default=3)
#     args = parser.parse_args()
