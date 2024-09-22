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
from utils.ddetr_utils import check_for_detections
from scipy.spatial import distance
import pickle

import skimage
from tqdm import tqdm
import glob
import copy
import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# from definitions.teach_objects import (
#     THING_NAMES,
#     STUFF_NAMES,
#     get_object_affordance,
#     get_object_receptacle_compatibility,
#     ObjectClass,
# )

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

        # cropped = rgb[x_min:x_max, y_min:y_max]

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


class AttributeDetectorImage():

    def __init__(
        self, 
        W, H, 
        ): 

        self.W, self.H = W, H

        from nets.clip import CLIP
        self.clip_model = CLIP()

        # self.root_eval_images = args.root_images_eval
        self.root_train_images = args.root_images_train

        self.images_train = [d for d in tqdm(glob.glob(self.root_train_images + '/attributes' + 4 * '/*'))]

        # self.images_eval = [d for d in tqdm(glob.glob(self.root_eval_images + '/attributes' + 4 * '/*'))]

        self.slicable_objects = ['Apple', 'Bread', 'Lettuce', 'Potato', 'Tomato']

        self.cookable_objects = ['Apple', 'Lettuce', 'Potato', 'Tomato', 'AppleSliced', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']

        self.toastable_objects = ['Bread', 'BreadSliced']

        self.cleanable_objects = ["Bowl", "Cup", "Mug", "Plate", "Pot", "Pan"]

        tmp_save_attribute_feature_dict = os.path.join(self.root_train_images, 'attribute_feature_dict.p')
        tmp_save_attribute_images_dict = os.path.join(self.root_train_images, 'attribute_images_dict.p')

        if os.path.exists(tmp_save_attribute_feature_dict):

            # Open the file in binary mode 
            with open(tmp_save_attribute_feature_dict, 'rb') as file: 
                # Call load method to deserialze 
                self.attribute_feature_dict = pickle.load(file) 

            # Open the file in binary mode 
            with open(tmp_save_attribute_images_dict, 'rb') as file: 
                # Call load method to deserialze 
                self.attribute_images_dict = pickle.load(file) 

        else:

            self.attribute_images_dict = {}
            count = 0
            for image in tqdm(self.images_train):
                split_image = image.split('/')
                category = split_image[-2]
                bool_val = split_image[-3]
                attribute = split_image[-4]
                if category not in self.attribute_images_dict.keys():
                    self.attribute_images_dict[category] = {}
                if attribute not in self.attribute_images_dict[category].keys():
                    self.attribute_images_dict[category][attribute] = {}
                if bool_val not in self.attribute_images_dict[category][attribute].keys():
                    self.attribute_images_dict[category][attribute][bool_val] = []
                img = Image.open(image)
                self.attribute_images_dict[category][attribute][bool_val].append(img.copy())
                img.close()
                count += 1
        
            self.attribute_feature_dict = copy.deepcopy(self.attribute_images_dict)
            for k1 in tqdm(self.attribute_images_dict.keys()):
                for k2 in self.attribute_images_dict[k1].keys():
                    for k3 in self.attribute_images_dict[k1][k2].keys():
                        with torch.no_grad():
                            image_features = self.clip_model.encode_images(self.attribute_images_dict[k1][k2][k3])
                        self.attribute_feature_dict[k1][k2][k3] = image_features.cpu()

            with open(tmp_save_attribute_images_dict, 'wb') as file: 
                # A new file will be created 
                pickle.dump(self.attribute_images_dict, file) 

            with open(tmp_save_attribute_feature_dict, 'wb') as file: 
                # A new file will be created 
                pickle.dump(self.attribute_feature_dict, file) 

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

    @torch.no_grad()
    def get_attributes(self, rgb, mask, object_category):

        box = self.mask_to_box(mask)
        x_min, x_max, y_min, y_max = box

        # cropped = rgb[x_min:x_max, y_min:y_max]

        cropped = rgb[y_min:y_max, x_min:x_max]

        probs = None

        if object_category in self.slicable_objects and object_category in self.attribute_images_dict.keys() and 'sliced' in self.attribute_images_dict[object_category].keys():
            attribute = 'sliced'
            bool_vals = sum([[k]*len(self.attribute_images_dict[object_category][attribute][k]) for k in self.attribute_images_dict[object_category][attribute].keys()], [])
            images_features = torch.cat([self.attribute_feature_dict[object_category][attribute][k] for k in self.attribute_feature_dict[object_category][attribute].keys()], axis=0)
            output = self.clip_model.score_images(cropped, images_features)
            argmax_output = torch.argmax(output)
            predicted_value = bool_vals[argmax_output]
            sliced = eval(predicted_value)
            # lines = [f"The {object_category} is sliced", f"The {object_category} is not sliced"]
            # probs = self.model.score(cropped, lines)[0]
            # sliced = True if probs[0]>probs[1] else False
            print(f'{object_category} sliced? {sliced}')
        else:
            print(f'Category {object_category} attribute sliced not detected')
            sliced = False

        if object_category in self.cookable_objects and object_category in self.attribute_images_dict.keys() and 'cooked' in self.attribute_images_dict[object_category].keys():
            attribute = 'cooked'
            bool_vals = sum([[k]*len(self.attribute_images_dict[object_category][attribute][k]) for k in self.attribute_images_dict[object_category][attribute].keys()], [])
            images_features = torch.cat([self.attribute_feature_dict[object_category][attribute][k] for k in self.attribute_feature_dict[object_category][attribute].keys()], axis=0)
            output = self.clip_model.score_images(cropped, images_features)
            argmax_output = torch.argmax(output)
            predicted_value = bool_vals[argmax_output]
            cooked = eval(predicted_value)
            # lines = [f"The {object_category} is cooked", f"The {object_category} is not cooked"]
            # probs = self.model.score(cropped, lines)[0]
            # cooked = True if probs[0]>probs[1] else False
            print(f'{object_category} cooked? {cooked}')
        else:
            print(f'Category {object_category} attribute cooked not detected')
            cooked = False

        if object_category in self.toastable_objects and object_category in self.attribute_images_dict.keys() and 'cooked' in self.attribute_images_dict[object_category].keys():
            attribute = 'cooked'
            bool_vals = sum([[k]*len(self.attribute_images_dict[object_category][attribute][k]) for k in self.attribute_images_dict[object_category][attribute].keys()], [])
            images_features = torch.cat([self.attribute_feature_dict[object_category][attribute][k] for k in self.attribute_feature_dict[object_category][attribute].keys()], axis=0)
            output = self.clip_model.score_images(cropped, images_features)
            argmax_output = torch.argmax(output)
            predicted_value = bool_vals[argmax_output]
            toasted = eval(predicted_value)
            # lines = [f"The {object_category} is toasted", f"The {object_category} is not toasted"]
            # probs = self.model.score(cropped, lines)[0]
            # toasted = True if probs[0]>probs[1] else False
            print(f'{object_category} toasted? {toasted}')
        else:
            print(f'Category {object_category} attribute toasted not detected')
            toasted = False

        if object_category in self.cleanable_objects and object_category in self.attribute_images_dict.keys() and 'dirty' in self.attribute_images_dict[object_category].keys():
            attribute = 'dirty'
            bool_vals = sum([[k]*len(self.attribute_images_dict[object_category][attribute][k]) for k in self.attribute_images_dict[object_category][attribute].keys()], [])
            images_features = torch.cat([self.attribute_feature_dict[object_category][attribute][k] for k in self.attribute_feature_dict[object_category][attribute].keys()], axis=0)
            output = self.clip_model.score_images(cropped, images_features)
            argmax_output = torch.argmax(output)
            predicted_value = bool_vals[argmax_output]
            clean = not eval(predicted_value)
            # lines = [f"The {object_category} is clean", f"The {object_category} is dirty"]
            # probs = self.model.score(cropped, lines)[0]
            # clean = True if probs[0]>probs[1] else False
            print(f'{object_category} clean? {clean}')
        else:
            print(f'Category {object_category} attribute clean not detected')
            clean = False

        # if object_category in self.cleanable_objects+self.toastable_objects+self.cookable_objects+self.slicable_objects:
        #     plt.figure()
        #     plt.imshow(rgb)
        #     plt.savefig('output/test1.png')
        #     plt.figure()
        #     plt.imshow(cropped)
        #     plt.savefig('output/test.png')
        #     print(sliced, cooked, toasted, clean)
        #     st()

        attributes = {
            "sliced":sliced,
            "toasted":toasted,
            "clean":clean,
            "cooked":cooked,
        }

        return attributes

class AttributeDetectorVisualMem():

    def __init__(
        self, 
        W, H, 
        ): 

        self.W, self.H = W, H

        from nets.clip import CLIP
        self.clip_model = CLIP()

        # self.root_eval_images = args.root_images_eval
        self.root_train_images = args.root_images_train

        self.images_train = [d for d in tqdm(glob.glob(self.root_train_images + '/attributes' + 4 * '/*'))]

        # self.images_eval = [d for d in tqdm(glob.glob(self.root_eval_images + '/attributes' + 4 * '/*'))]

        self.attributes_to_check = ['cooked', 'dirty']

        self.slicable_objects = ['Apple', 'Bread', 'Lettuce', 'Potato', 'Tomato']

        self.cookable_objects = ['Apple', 'Lettuce', 'Potato', 'Tomato', 'AppleSliced', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']

        self.toastable_objects = ['Bread', 'BreadSliced']

        self.cleanable_objects = ["Bowl", "Cup", "Mug", "Plate", "Pot", "Pan"]

        self.score_type = 'max' # generally max works the best

        tmp_save_attribute_feature_dict = os.path.join(self.root_train_images, f'attribute_feature_dict_{args.clip_model.replace("/", "-").replace("-", "_")}.p')
        tmp_save_attribute_images_dict = os.path.join(self.root_train_images, f'attribute_images_dict_{args.clip_model.replace("/", "-").replace("-", "_")}.p')

        if os.path.exists(tmp_save_attribute_feature_dict):

            # Open the file in binary mode 
            with open(tmp_save_attribute_feature_dict, 'rb') as file: 
                # Call load method to deserialze 
                self.attribute_feature_dict = pickle.load(file) 

            # Open the file in binary mode 
            with open(tmp_save_attribute_images_dict, 'rb') as file: 
                # Call load method to deserialze 
                self.attribute_images_dict = pickle.load(file) 

        else:

            self.attribute_images_dict = {}
            count = 0
            for image in tqdm(self.images_train):
                split_image = image.split('/')
                category = split_image[-2]
                bool_val = split_image[-3]
                attribute = split_image[-4]
                if attribute not in self.attributes_to_check:
                    continue
                if category not in self.attribute_images_dict.keys():
                    self.attribute_images_dict[category] = {}
                if attribute not in self.attribute_images_dict[category].keys():
                    self.attribute_images_dict[category][attribute] = {}
                if bool_val not in self.attribute_images_dict[category][attribute].keys():
                    self.attribute_images_dict[category][attribute][bool_val] = []
                img = Image.open(image)
                self.attribute_images_dict[category][attribute][bool_val].append(img.copy())
                img.close()
                count += 1
        
            self.attribute_feature_dict = copy.deepcopy(self.attribute_images_dict)
            for k1 in tqdm(self.attribute_images_dict.keys()):
                for k2 in self.attribute_images_dict[k1].keys():
                    for k3 in self.attribute_images_dict[k1][k2].keys():
                        with torch.no_grad():
                            image_features = self.clip_model.encode_images(self.attribute_images_dict[k1][k2][k3])
                        self.attribute_feature_dict[k1][k2][k3] = image_features.cpu()

            with open(tmp_save_attribute_images_dict, 'wb') as file: 
                # A new file will be created 
                pickle.dump(self.attribute_images_dict, file) 

            with open(tmp_save_attribute_feature_dict, 'wb') as file: 
                # A new file will be created 
                pickle.dump(self.attribute_feature_dict, file) 

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


    @torch.no_grad()
    def check_attribute(self, rgb, mask, category, attribute):

        invert = False
        if attribute=="clean":
            invert = True
            attribute = "dirty"

        if category not in self.attribute_images_dict.keys() or attribute not in self.attribute_images_dict[category].keys():
            return None

        box = self.mask_to_box(mask, padding=0)
        # x_min, x_max, y_min, y_max = box

        box = box[[0,2,1,3]]

        max_width = max(box[2] - box[0], box[3] - box[1])
                
        padding = int(np.ceil(max_width * 0.1))
        box[[0,1]] -= padding
        box[[2,3]] += padding
        x_min, y_min, x_max, y_max = list(np.clip(box, a_min=0, a_max=self.W-1))
        # x_min, x_max, y_min, y_max = self.mask_to_box(i_mask, padding=int(self.W*0.05))
        cropped = rgb[y_min:y_max, x_min:x_max]

        # cropped = rgb[x_min:x_max, y_min:y_max]

        # cropped = rgb[y_min:y_max, x_min:x_max]

        bool_vals = sum([[k]*len(self.attribute_images_dict[category][attribute][k]) for k in self.attribute_images_dict[category][attribute].keys()], [])
        # images = sum([self.attribute_images_dict[category][attribute][k] for k in self.attribute_images_dict[category][attribute].keys()], [])
        images_features = torch.cat([self.attribute_feature_dict[category][attribute][k] for k in self.attribute_feature_dict[category][attribute].keys()], axis=0)
        img = Image.fromarray(cropped)
        # img = Image.open(image)
        image_test = [img.copy()]
        img.close()
        output = self.clip_model.score_images(image_test, images_features)
        # output[output==1] = 0.
        if self.score_type=='max':
            argmax_output = torch.argmax(output)
            # argmax_output = torch.argmin(output)
            predicted_value = eval(bool_vals[argmax_output])
            # bool_val = eval(bool_val)
        elif self.score_type=='topk_mean':
            k = 5
            bool_vals_np = np.asarray(bool_vals)
            unique_vals = np.unique(bool_vals_np, return_index=False, return_inverse=False, return_counts=False)
            mean_vals = torch.zeros(len(unique_vals)).to(output.device)
            for val_idx in range(len(unique_vals)):
                val = unique_vals[val_idx]
                mask = bool_vals_np==val
                masked_output = output[mask]
                masked_output = torch.topk(masked_output, k).values
                mean_vals[val_idx] = torch.mean(masked_output)
            argmax_output = torch.argmax(mean_vals)
            predicted_value = bool_vals[argmax_output]
            predicted_value = eval(bool_vals[argmax_output])
            # bool_val = eval(bool_val)
        elif self.score_type=='mean':
            bool_vals_np = np.asarray(bool_vals)
            unique_vals = np.unique(bool_vals_np, return_index=False, return_inverse=False, return_counts=False)
            mean_vals = torch.zeros(len(unique_vals)).to(output.device)
            for val_idx in range(len(unique_vals)):
                val = unique_vals[val_idx]
                mask = bool_vals_np==val
                masked_output = output[mask]
                mean_vals[val_idx] = torch.mean(masked_output)
            argmax_output = torch.argmax(mean_vals)
            predicted_value = bool_vals[argmax_output]
            predicted_value = eval(bool_vals[argmax_output])
            # bool_val = eval(bool_val)
        else:
            raise NotImplementedError

        if invert:
            predicted_value = not predicted_value

        return predicted_value