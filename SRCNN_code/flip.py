# -*- coding: utf-8 -*-
import torch
# Implementation of geometric self-ensamble as seen in the EDSR paper 

# import torch.backends.cudnn as cudnn
# from models import SRCNN
# import PIL.Image as pil_image
from PIL import ImageOps 
from utils import preprocess, preprocess2, from_pred_to_image
# import numpy as np

def flip_im(image, model, y, device):
    # image_file="data/Cars/data_ran/y_1.bmp"
    # image = pil_image.open(image_file)
    
    angle = 90
    out1 = image.rotate(angle)
    out1, y1 = preprocess(out1, device)
    out2 = image.rotate(angle*2)
    out2, y2 = preprocess(out2, device)
    out3 = image.rotate(-angle)
    out3, y3 = preprocess(out3, device)
    out4 = ImageOps.mirror(image)
    out4, y4 = preprocess(out4, device)
    # 4 plus y
    # array the images but I dont know how to undo the rotation 
    
    with torch.no_grad():
         pred1 = model(y).clamp(0.0, 1.0)
         
    with torch.no_grad():
         pred2 = model(out1).clamp(0.0, 1.0)
         # convertir otra vez a un objeto pil image para poder rotarla coon la misma fucion hacer la rotacio inversa
         # convertir otra vez a array para hacer el promedio y el resto de los cálculos 
         
         # im_pred2 = pil_image.fromarray(np.uint8(pred2)).convert('RGB')
         # im_pred2 = pil_image.fromarray(pred2.astype('uint8'), 'RGB')
         # pred2 = pred2.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
         # im_pred2 = np.array([pred2, y1[..., 1], y1[..., 2]]).transpose([1, 2, 0])
         # im_pred2 = np.clip(convert_ycbcr_to_rgb(im_pred2), 0.0, 255.0).astype(np.uint8)
         # im_pred2 = pil_image.fromarray(im_pred2)
         im_pred2 = from_pred_to_image(pred2, y1)
         im_pred2 = im_pred2.rotate(-angle)
         pred2, _ = preprocess2(im_pred2, device)
         
    with torch.no_grad():
         pred3 = model(out2).clamp(0.0, 1.0)
         im_pred3 = from_pred_to_image(pred3, y2)
         im_pred3 = im_pred3.rotate(-angle*2)
         pred3, _ = preprocess2(im_pred3, device)
         
    with torch.no_grad():
         pred4 = model(out3).clamp(0.0, 1.0)
         im_pred4 = from_pred_to_image(pred4, y3)
         im_pred4 = im_pred4.rotate(angle)
         pred4, _ = preprocess2(im_pred4, device)
         
    with torch.no_grad():
         pred5 = model(out4).clamp(0.0, 1.0)
         im_pred5 = from_pred_to_image(pred5, y4)
         im_pred5 = ImageOps.mirror(im_pred5)
         pred5, _ = preprocess2(im_pred5, device)
    
    pred_a=(pred1+pred2+pred3+pred4+pred5)/5
    
    return pred_a
    # out1.show()
    # out2.show()
    # out3.show()
    # out4.show()
