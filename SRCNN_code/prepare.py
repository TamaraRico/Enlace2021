import argparse
import glob
import h5py
import numpy as np
#import os 
import PIL.Image as pil_image
from utils import convert_rgb_to_y
from tqdm import trange

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    # for image_path in sorted(glob.glob('{}/x_*[0-999]*'.format(args.images_dir))):
    #     hr = pil_image.open(image_path).convert('RGB')
    #     # hr = hr.show()
    #     #hr.save('color_.bmp')
    #     # hr_width = (hr.width // args.scale) * args.scale
    #     # hr_height = (hr.height // args.scale) * args.scale
    #     # hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
    #     # lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
    #     # lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    #     # lr = pil_image.open(image_path).convert('RGB')
    #     hr = np.array(hr).astype(np.float32)
    #     # lr = np.array(lr).astype(np.float32)
    #     hr = convert_rgb_to_y(hr)
    #     # lr = convert_rgb_to_y(lr)
        
    #     # #modify by Tamara
    #     for i in range(0, hr.shape[0] - args.patch_size + 1, args.stride):
    #         for j in range(0, hr.shape[1] - args.patch_size + 1, args.stride):
    #             hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])
        
    # for image_path in sorted(glob.glob('{}/y_*[0-999]*'.format(args.images_dir))):
    #     lr = pil_image.open(image_path).convert('RGB')
    #     lr = np.array(lr).astype(np.float32)
    #     lr = convert_rgb_to_y(lr)

    #     for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
    #         for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
    #             lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
    #                     # hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])
    tr_number_images = args.number_imgs
    #for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
    for i in trange(tr_number_images):
        # changed from pgn to bmp due to Jake'comments about save img 03/ago/2021
        fileX = args.images_dir + "/x_" + str(i) + ".bmp"
        hr = pil_image.open(fileX).convert('RGB')

        fileY = args.images_dir + "/y_" + str(i) + ".bmp"
        lr = pil_image.open(fileY).convert('RGB')

        #hr_width = (hr.width // args.scale) * args.scale
        #hr_height = (hr.height // args.scale) * args.scale
        #hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        #lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        #lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    # if not os.path.exists("Training"):
    #     os.makedirs("Training")
    
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    
    ev_number_images = args.number_imgs

    #for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
    for i in trange(ev_number_images):
        fileX = args.images_dir + "/x_" + str(i) + ".bmp"
        hr = pil_image.open(fileX).convert('RGB')

        fileY = args.images_dir + "/y_" + str(i) + ".bmp"
        lr = pil_image.open(fileY).convert('RGB')

        #hr_width = (hr.width // args.scale) * args.scale
        #hr_height = (hr.height // args.scale) * args.scale
        #hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        #lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        #lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

    # for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
    #     hr = pil_image.open(image_path).convert('RGB')
    #     hr_width = (hr.width // args.scale) * args.scale
    #     hr_height = (hr.height // args.scale) * args.scale
    #     hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
    #     lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
    #     lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    #     hr = np.array(hr).astype(np.float32)
    #     lr = np.array(lr).astype(np.float32)
    #     hr = convert_rgb_to_y(hr)
    #     lr = convert_rgb_to_y(lr)

        hr_group.create_dataset(str(i), data=hr)
        lr_group.create_dataset(str(i), data=lr)

        
    # for i, image_path in enumerate(sorted(glob.glob('{}/y_*[0-999]*'.format(args.images_dir)))):
    #     lr = pil_image.open(image_path).convert('RGB')
    #     # hr_group.create_dataset(str(i), data=hr)
    #     lr = np.array(lr).astype(np.float32)
    #     lr = convert_rgb_to_y(lr)

    #     lr_group.create_dataset(str(i), data=lr)


    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    # added by Jose
    parser.add_argument('--number-imgs', type=int, required=True)
    parser.add_argument('--patch-size', type=int, default=33) #the size of the image
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    print(args.eval)

    if not args.eval:
        train(args)

    else:
        eval(args)
