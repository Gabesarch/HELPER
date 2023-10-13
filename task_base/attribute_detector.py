import numpy as np
import utils.aithor
import torch
from PIL import Image
import ipdb
st = ipdb.set_trace
from utils.ddetr_utils import check_for_detections
import numpy as np
import utils.aithor
import utils.geom
import torch
from PIL import Image
from arguments import args
import sys
from backend import saverloader
import os
import cv2
import matplotlib.pyplot as plt
import ipdb
st = ipdb.set_trace
from scipy.spatial import distance

import skimage

class AttributeDetector():

    def __init__(
        self, 
        W, H, 
        ): 

        self.W, self.H = W, H

        from nets.clip import ALIGN
        self.model = ALIGN()

        self.slicable_objects = ['Apple', 'Bread', 'Lettuce', 'Potato', 'Tomato']

        self.cookable_objects = ['Apple', 'Lettuce', 'Potato', 'Tomato', 'AppleSliced', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']

        self.toastable_objects = ['Bread', 'BreadSliced']

        self.cleanable_objects = ["Bowl", "Cup", "Mug", "Plate", "Pot", "Pan"]

    def mask_to_box(self, mask, padding = 20):

        segmentation = np.where(mask == True)

        # Bounding Box
        bbox = np.asarray([0, 0, 0, 0])
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            bbox = np.asarray([max(0, x_min-padding), min(x_max+padding, self.W-1), max(0, y_min-padding), min(y_max+padding, self.H-1)])

        return bbox

    def get_attributes(self, rgb, mask, object_category):

        box = self.mask_to_box(mask)
        x_min, x_max, y_min, y_max = box

        cropped = rgb[y_min:y_max, x_min:x_max]

        probs = None

        if object_category in self.slicable_objects:
            lines = [f"The {object_category} is sliced", f"The {object_category} is not sliced"]
            probs = self.model.score(cropped, lines)[0]
            sliced = True if probs[0]>probs[1] else False
            print(f'{object_category} sliced? {sliced}')
        else:
            sliced = False

        if object_category in self.cookable_objects:
            lines = [f"The {object_category} is cooked", f"The {object_category} is not cooked"]
            probs = self.model.score(cropped, lines)[0]
            cooked = True if probs[0]>probs[1] else False
            print(f'{object_category} cooked? {cooked}')
        else:
            cooked = False

        if object_category in self.toastable_objects:
            lines = [f"The {object_category} is toasted", f"The {object_category} is not toasted"]
            probs = self.model.score(cropped, lines)[0]
            toasted = True if probs[0]>probs[1] else False
            print(f'{object_category} toasted? {toasted}')
        else:
            toasted = False

        if object_category in self.cleanable_objects:
            lines = [f"The {object_category} is clean", f"The {object_category} is dirty"]
            probs = self.model.score(cropped, lines)[0]
            clean = True if probs[0]>probs[1] else False
            print(f'{object_category} clean? {clean}')
        else:
            clean = False

        # if probs is not None:
        #     # visualize crops
        #     plt.figure()
        #     plt.imshow(rgb)
        #     plt.savefig('output/images/test1.png')
        #     plt.figure()
        #     plt.imshow(cropped)
        #     plt.savefig('output/images/test.png')
        #     print(sliced, cooked, toasted, clean)
        #     st()

        attributes = {
            "sliced":sliced,
            "toasted":toasted,
            "clean":clean,
            "cooked":cooked,
        }

        return attributes