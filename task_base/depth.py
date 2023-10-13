import numpy as np
from collections import Counter, OrderedDict
from arguments import args
import torch
from torchvision import transforms
from PIL import Image
import cv2
import ipdb
st = ipdb.set_trace
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Depth_ZOE():
    def __init__(self, estimate_depth=True, task=None, on_aws=False, DH=300, DW=300):
        from nets.depthnet import DepthNet

        self.task = task
        self.agent = self.task

        self.W = args.W
        self.H = args.H

        self.DH = DH
        self.DW = DW

        self.args = args

        self.estimate_depth = estimate_depth
        self.use_learned_depth = estimate_depth

        self.max_depth = 10.0
        self.min_depth = 0.01


        if self.estimate_depth:

            self.model = DepthNet(pretrained=False)

            if args.zoedepth_checkpoint is not None:
                checkpoint = torch.load(args.zoedepth_checkpoint)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

            self.model.to(device)
            self.model.eval()

    @torch.no_grad()
    def get_depth_map(self, rgb, head_tilt, filter_depth_by_sem=False):
        
        if self.estimate_depth:
            rgb_norm = rgb.astype(np.float32) * 1./255
            rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
            depth = self.model(rgb_torch)
            depth = torch.nn.functional.interpolate(
                                    depth,
                                    size=(self.W, self.H),
                                    mode="nearest",
                                    # align_corners=False,
                                ).squeeze().cpu().numpy()

            depth_invalid = np.logical_or(depth>self.max_depth, depth<self.min_depth)
            depth[depth_invalid] = np.nan
            sem_seg_pred = None
        else:
            depth = self.task.env.simulator.controller.last_event.depth_frame
            sem_seg_pred = None

        return depth, sem_seg_pred