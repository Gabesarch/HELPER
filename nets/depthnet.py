import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from torchvision import models
import ipdb
st = ipdb.set_trace
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append("ZoeDepth")
from zoedepth.models.zoedepth import ZoeDepth
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import GradL1Loss, SILogLoss
from zoedepth.models.model_io import load_state_from_resource, load_wts
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from arguments import args

# fix the seed for reproducibility
seed = args.seed 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# def preprocessing_transforms(mode, **kwargs):
#     return transforms.Compose([
#         ToTensor(mode=mode, **kwargs)
#     ])

class DepthNet(nn.Module):
    def __init__(self, do_normalize=False, size=None, pretrained=True):
        super(DepthNet, self).__init__()

        model = "zoedepth"
        dataset = "nyu"

        overwrite_kwargs = {"model":model}

        config = get_config(model, "train", dataset, **overwrite_kwargs)
        config.batch_size = args.batch_size
        config.mode = 'train'
        config.input_height = args.H
        config.input_width = args.W
        self.config = config

        self.model = build_model(config)

        if pretrained:
            if not os.path.exists(os.path.join(args.torch_checkpoint_path, 'ZoeD_M12_N.pt')):
                # download checkpoint
                torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
                repo = "isl-org/ZoeDepth"
                # Zoe_N
                torch.hub.load(repo, "ZoeD_N", pretrained=True)
            self.model = load_wts(self.model, os.path.join(args.torch_checkpoint_path, 'ZoeD_M12_N.pt'))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size)
        else:
            self.resize = nn.Identity()
        
    def forward(self, rgb):

        rgb = self.normalize(rgb)
        rgb = self.resize(rgb)

        out = self.model(rgb)

        metric_depth = out['metric_depth']

        return metric_depth

class ZoeDepthLoss(nn.Module):
    def __init__(self, config):
        super(ZoeDepthLoss, self).__init__()

        self.silog_loss = SILogLoss()
        self.grad_loss = GradL1Loss()
        self.config = config

    def forward(self, pred_depths, depths_gt, mask=None):

        losses = {}
        total_loss = torch.tensor(0.0).to(device)

        l_si, pred = self.silog_loss(
            pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
        loss = self.config.w_si * l_si
        losses['silog_loss'] = l_si
        total_loss += loss
        # losses[self.silog_loss.name] = l_si

        if self.config.w_grad > 0:
            l_grad = self.grad_loss(pred, depths_gt, mask=mask)
            total_loss = total_loss + self.config.w_grad * l_grad
            losses['l_grad'] = l_grad
        else:
            l_grad = torch.tensor(0.0).to(device)
            losses['l_grad'] = l_grad
        
        return total_loss, losses, pred
        

# class ToTensor(object):
#     def __init__(self, mode, do_normalize=False, size=None):
#         self.mode = mode
#         self.normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
#         self.size = size
#         if size is not None:
#             self.resize = transforms.Resize(size=size)
#         else:
#             self.resize = nn.Identity()

#     def __call__(self, sample):
#         image, focal = sample['image'], sample['focal']
#         image = self.to_tensor(image)
#         image = self.normalize(image)
#         image = self.resize(image)

#         if self.mode == 'test':
#             return {'image': image, 'focal': focal}

#         depth = sample['depth']
#         if self.mode == 'train':
#             depth = self.to_tensor(depth)
#             return {**sample, 'image': image, 'depth': depth, 'focal': focal}
#         else:
#             has_valid_depth = sample['has_valid_depth']
#             image = self.resize(image)
#             return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
#                     'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

#     def to_tensor(self, pic):
#         if not (_is_pil_image(pic) or _is_numpy_image(pic)):
#             raise TypeError(
#                 'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

#         if isinstance(pic, np.ndarray):
#             img = torch.from_numpy(pic.transpose((2, 0, 1)))
#             return img

#         # handle PIL Image
#         if pic.mode == 'I':
#             img = torch.from_numpy(np.array(pic, np.int32, copy=False))
#         elif pic.mode == 'I;16':
#             img = torch.from_numpy(np.array(pic, np.int16, copy=False))
#         else:
#             img = torch.ByteTensor(
#                 torch.ByteStorage.from_buffer(pic.tobytes()))
#         # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#         if pic.mode == 'YCbCr':
#             nchannel = 3
#         elif pic.mode == 'I;16':
#             nchannel = 1
#         else:
#             nchannel = len(pic.mode)
#         img = img.view(pic.size[1], pic.size[0], nchannel)

#         img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         if isinstance(img, torch.ByteTensor):
#             return img.float()
#         else:
#             return img