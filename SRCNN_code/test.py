import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

import pytorch_ssim
import pytorch_msssim

from flip import flip_im
from models import SRCNN
from utils import convert_ycbcr_to_rgb, calc_psnr, preprocess,preprocess2, get_BI_image


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights-file', type=str, required=True)
    # parser.add_argument('--image-file', type=str, required=True)
    # parser.add_argument('--scale', type=int, default=3)
    # args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)
    
    # to test the SRCNN in the data set created using point-source data
    num_img=100

    state_dict = model.state_dict()
    # weights_file="data/Training/x3/bestAll.pth"
    
    # Jose pht file
    weights_file="data/Training/x3/bestAll.pth"
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    # i=get_i() WRONG !!
    
    # acumuladores 
    psnr_a=0
    psnr1_a=0
    ssim_a=0
    ssim1_a=0
    msssim_a=0
    msssim1_a=0
     
    scale=3
    # this for loop is used test the SRCNN and see how performes in this point source data
    # there was another way to do it but I haven't had luck with the code 
    # si lo vemos en clase, corregir
    for i in range(num_img):
        
        # here start the changes I made 
        image_file="data/Cars/data_x3/y_"+str(i)+".bmp"
        true_image_file ="data/Cars/data_x3/x_"+str(i)+".bmp"
        # image_file="data/Cars/data/y_"+str(i)+".png"
        # true_image_file ="data/Cars/data/x_"+str(i)+".png"
        #image_file="data/Cars/data/x_0.png"
        image = pil_image.open(image_file).convert('RGB')
        # image = pil_image.open(args.image_file).convert('RGB')
        true_image = pil_image.open(true_image_file).convert('RGB')# [::scale, ::scale]
        
        x_img_w = (true_image.width // scale) * scale
        x_img_h = (true_image.height // scale) * scale
        true_image = true_image.resize((x_img_w, x_img_h))
        true_image = true_image.resize((true_image.width // scale, true_image.height // scale))
        true_image = true_image.resize((true_image.width * scale, true_image.height * scale))
        # print(true_image.size()) no se puede aqui

        # to see if I can upsample the image
        # this was remove before
        image_width = (image.width // scale) * scale
        image_height = (image.height // scale) * scale
        image = image.resize((image_width, image_height))#, resample=pil_image.BICUBIC)
        image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
        image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
        # print(image.size())
        
        bicubic=get_BI_image(image, scale, image_file, device)
        # print(bicubic.size())
    
        image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)

        # by Jose
        #scale=3 # by default 
        
        # imageO = image
    
        # image_widthO = (imageO.width // args.scale) * args.scale
        # image_heightO = (imageO.height // args.scale) * args.scale
        # imageO = image.resize((image_widthO, image_heightO))
        # imageO = image.resize((imageO.width // args.scale, imageO.height // args.scale))
        # imageO = image.resize((imageO.width * args.scale, imageO.height * args.scale))
        
        # image_lr = image.resize((imageO.width // scale, imageO.height // scale))
        # image_lr.save(image_file.replace('.', f'_scale_x{scale}.'))
        
        # image_widthO = (imageO.width // scale) * scale
        # image_heightO = (imageO.height // scale) * scale
        # imageO = image.resize((image_widthO, image_heightO))
        # imageO = image.resize((imageO.width // scale, imageO.height // scale))
        # imageO = image.resize((imageO.width * scale, imageO.height * scale))         
        
        # image_width = (image.width // args.scale) * args.scale
        # image_height = (image.height // args.scale) * args.scale
        # image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        
        # Fist try to get the bicubic image / WRONG
        # im_bi=pil_image.BICUBIC
        # im_bi=np.array(im_bi).astype(np.float32)
        # im1 = convert_rgb_to_ycbcr(im)
        # x = im1[..., 0]
        # x /= 255.
        # x = torch.from_numpy(x).to(device)
        # x = x.unsqueeze(0).unsqueeze(0)
        # END OF FIRST TRY 
        
        # image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        # image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
        
        # image_or = image.resize((image_or.width // args.scale, image_or.height // args.scale))
        # image_or = image.resize((image_or.width * args.scale, image_or.height * args.scale))
        
        # image = np.array(image).astype(np.float32)
        # ycbcrBC = convert_rgb_to_ycbcr(image)
        # y = ycbcrBC[..., 0]
        # y /= 255.
        # y = torch.from_numpy(y).to(device)
        # y =  y.unsqueeze(0).unsqueeze(0)
    
        # image_width = (image.width // args.scale) * args.scale
        # image_height = (image.height // args.scale) * args.scale
        # image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        # image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
        # image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
        # image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
    
        # image_width = (image.width // scale) * scale
        # image_height = (image.height // scale) * scale
        # image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        # image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
        # image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
        # image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))         
    
        # SECOND TRY / still wrong
        # to get the bicubic image to the metrics 
        # im_bi=pil_image.BICUBIC
        # print(im_bi)
        # im_bi=np.array(im_bi).astype(np.float32)
        # im1 = convert_rgb_to_ycbcr(im_bi)
        # x = im1[..., 0]
        # x /= 255.
        # x = torch.from_numpy(x).to(device)
        # print(x)
        # end 
        
        # image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
        # image = np.array(image).astype(np.float32)
        # bic=torch.from_numpy(image).to(device)
        # bic=bic.unsqueeze(0).unsqueeze(0)
        # ycbcr = convert_rgb_to_ycbcr(image)
        
        # this is the high resolution image, needed to compare the output of the SRCNN
        # timage_width = (true_image.width // scale) * scale
        # timage_height = (true_image.height // scale) * scale
        # true_image = image.resize((timage_width, timage_height))
        # true_image = image.resize((true_image.width // scale, true_image.height // scale))
        # true_image = image.resize((true_image.width * scale, true_image.height * scale))
        
        # PASS TO A FUNCTION IN UTILS
        # true_image = np.array(true_image).astype(np.float32)[::2, ::2]# [::4, ::4]#[::scale, ::scale]
        # ycbcr = convert_rgb_to_ycbcr(true_image)
        # t_img = ycbcr[..., 0]
        # t_img /= 255.
        # t_img = torch.from_numpy(t_img).to(device)
        # t_img = t_img.unsqueeze(0).unsqueeze(0)
        t_img, ycbcr = preprocess(true_image, device)
        # print(t_img.size())
        
        # this is the low resolution image
        # img_width = (image.width // scale) * scale
        # img_height = (image.height // scale) * scale
        # image = image.resize((img_width, img_height), resample=pil_image.BICUBIC)
        # image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
        # image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
        #y_img.save(fileY.replace('.', '_bicub_x'+str(scale_factor)+'.'))
        
        # MOVE TO THE FUNCTION preprocess IN UTILS
        # image = np.array(image).astype(np.float32)[::2, ::2]#[::4, ::4]
        # ycbcr = convert_rgb_to_ycbcr(image)
        # y = ycbcr[..., 0]
        # y /= 255.
        # y = torch.from_numpy(y).to(device)
        # y = y.unsqueeze(0).unsqueeze(0)
        y, ycbcr = preprocess2(image, device)
        # print(y.size())
    
        # trying to undertand what i was doing
        # function given by Jake
        # print('max y:  ', y)#np.amax(y)) mthhis isnt a numpy array, is a tensor 
        # print('max lim_bicubic:  ') # , im_bi)# np.amax(im_bi)) #this is numoy so the function worked 
        # end 
    
        # with torch.no_grad():
        #     preds = model(y).clamp(0.0, 1.0)
        preds = flip_im(image, model, y, device)
        # print(preds.size())
        
        psnr1 = calc_psnr(t_img, bicubic)
        psnr1 = calc_psnr(t_img, bicubic)
        psnr1_a+=psnr1
        ssim1_value = pytorch_ssim.ssim(t_img, bicubic)
        ssim1_a+=ssim1_value
        msssim1_value = pytorch_msssim.msssim(t_img, bicubic)
        msssim1_a+=msssim1_value
        
        psnr = calc_psnr(t_img, preds)
        psnr_a+=psnr
        ssim_value = pytorch_ssim.ssim(t_img, preds)
        ssim_a+=ssim_value
        msssim_value = pytorch_msssim.msssim(t_img, preds)
        msssim_a+=msssim_value
        
        # preds = preds/preds.max()
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(image_file.replace('.', '_srcnntrial_x'+str(scale)+'.'.format(scale)))
        # output.save(image_file.replace('.', '_srcnn_x{}.'.format(scale)))

    # DATA FROM THE BICUBIC and ORIGINAL
    print('Bibubic vs Original')
    #psnr1 = calc_psnr(y, iBC)
    #print('PSNR: {:.2f}'.format(psnr1))
    print('PSNR: {:.2f}'.format(psnr1_a/num_img))
    
    # ssim1_value = pytorch_ssim.ssim(y, iBC)
    # print('SSIM_Zhou Wang: {:.2f}'.format(ssim1_value))
    print('SSIM_Zhou Wang: {:.2f}'.format(ssim1_a/num_img))
    
    # msssim1_value = pytorch_msssim.msssim(y, iBC)
    # print('MsSSIM: {:.2f}'.format(msssim1_value))
    print('MsSSIM: {:.2f}'.format(msssim1_a/num_img))
    print('\n')

    # DATA FROM THE SRCNN vs ORIGINAL
    print('SRCNN vs Original')
    # psnr = calc_psnr(y, preds)
    # print('PSNR: {:.2f}'.format(psnr))
    print('PSNR: {:.2f}'.format(psnr_a/num_img))
    
    # ssim_value = pytorch_ssim.ssim(y, preds)
    # print('SSIM_Zhou Wang: {:.2f}'.format(ssim_value))
    print('SSIM_Zhou Wang: {:.2f}'.format(ssim_a/num_img))
    
    # msssim_value = pytorch_msssim.msssim(y, preds)
    # print('MSSSIM: {:.2f}'.format(msssim_value))
    print('MSSSIM: {:.2f}'.format(msssim_a/num_img))
    print('\n')

    # preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    # output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    # output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    # output = pil_image.fromarray(output)
    # output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))
    # output.save(image_file.replace('.', '_srcnn_x{}.'.format(scale)))
