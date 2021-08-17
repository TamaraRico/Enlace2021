import torch
import numpy as np
import torch.nn.functional as F
import PIL.Image as pil_image
# import pytorch_ssim

# by me, to make the code more simple
def get_BI_image(img, scale, image_file, device):
    # img = img.resize((img.width // scale, img.height // scale), resample=pil_image.BICUBIC)
    # img = img.resize((img.width // scale, img.height // scale), resample=pil_image.BICUBIC)
    img = img.resize((img.width * scale, img.height * scale), resample=pil_image.BICUBIC)
    img.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))
    imageBC = np.array(img).astype(np.float32)#[::6, ::6]# [::8, ::8]#[::4, ::4]
    ycbcrBC = convert_rgb_to_ycbcr(imageBC)
    iBC = ycbcrBC[..., 0]
    iBC /= 255.
    iBC = torch.from_numpy(iBC).to(device)
    iBC =  iBC.unsqueeze(0).unsqueeze(0)
    return iBC

# by me, because the code was repetitive
def from_pred_to_image(pred, yc):
    pred = pred.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    im_pred = np.array([pred, yc[..., 1], yc[..., 2]]).transpose([1, 2, 0])
    im_pred = np.clip(convert_ycbcr_to_rgb(im_pred), 0.0, 255.0).astype(np.uint8)
    im_pred = pil_image.fromarray(im_pred)
    return im_pred

def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

# wpsnr value by me 
def calc_wpsnr(img1, img2):
    # return 10. * torch.log10(1. / (1./1+(torch.std(img1, unbiased=False))**2)*(torch.mean((img1 - img2) ** 2)))
    return 10. * torch.log10(1.*(1+(torch.std(img1, unbiased=False))**2) / (torch.mean((img1 - img2) ** 2)))

def calc_nqm(img1, img2):
    # return 10. * torch.log10(1.*(torch.sum(img2)**2)./ torch.mean((img1 - img2) ** 2))
    # return 10. * torch.log10(torch.sum(img1)**2. / torch.mean((img1 - img2) ** 2))
    return 10. * torch.log10(torch.sum((img1).pow(2)) / torch.sum((img1 - img2) ** 2))
    # torch.sum

# SSIM function by me 
# def calc_ssim(img1, img2):
    # ssim=pytorch_ssim.ssim(img1, img2)
    # ssim_loss = pytorch_ssim.SSIM(window_size = 11)
    # return ssim_loss(img1, img2)
#     return torch.SSIM(kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True)

# IFC by me
def calc_ifc(img1, img2):
    return torch.sqrt((torch.mean((img1 - img2) ** 2)))
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    # return  loss = loss_fn(img2, img1)

# code from https://www.programmersought.com/article/11465221470/
def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)
    
def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False, weights=None):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val
# end of the code from the previous source

# by https://github.com/yjn870/FSRCNN-pytorch/blob/master/utils.py, to make the code less repetitve
def preprocess(img, device):
    img = np.array(img).astype(np.float32)#[::2, ::2]#[::4, ::4]#[::2, ::2]
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr

# el doble downsampling hacia que no se pudiera realizar el resto del codigo
# buscar otra solucion, having to practical identical functions makes no sense
def preprocess2(img, device):
    img = np.array(img).astype(np.float32)#[::3, ::3]#[::2, ::2]
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
