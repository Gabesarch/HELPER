import numpy as np
import cv2
import ipdb
st = ipdb.set_trace
from matplotlib import pyplot as plt 
from task_base.aithor_base import Base
import os
import requests
from PIL import Image
import torch
import sys
​
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
​
# segment anything
from segment_anything import build_sam, SamPredictor 
​
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
​
class OpenSetSegmenter():
​
    def __init__(self, categories, name_to_parsed_name, name_to_id):
​
        print("initializing grounding dino and segment everything...")
​
        # cfg
        config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py' #args.config  # change the path of the model config file
        grounded_checkpoint = '/projects/katefgroup/REPLAY/checkpoints/Grounded-Segment-Anything/groundingdino_swint_ogc.pth' #args.grounded_checkpoint  # change the path of the model
        sam_checkpoint = '/projects/katefgroup/REPLAY/checkpoints/Grounded-Segment-Anything/sam_vit_h_4b8939.pth' #args.sam_checkpoint
        self.image_path = '/home/gsarch/repo/RICK/Grounded_Segment_Anything/assets/demo1.jpg' #args.input_image
        self.box_threshold = 0.4 #args.box_threshold
        self.text_threshold = 0.35 #args.box_threshold
​​
        self.text_prompt = '. '.join([name_to_parsed_name[c.lower()] if c.lower() in name_to_parsed_name.keys() else c.lower() for c in categories])
​
        self.parsed_name_to_name = {}
        for c in categories:
            if c.lower() in name_to_parsed_name.keys():
                self.parsed_name_to_name[name_to_parsed_name[c.lower()]] = c  
            else:
                self.parsed_name_to_name[c.lower()] = c
        self.name_to_id = name_to_id
        self.grounder = load_model(config_file, grounded_checkpoint, device=device)

​
        # initialize SAM
        self.segmenter = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        self.segmenter.model.to(device)
​
        self.count = 0
​
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # self.segmenter.set_image(image)
​
        # size = image_pil.size
        # H, W = size[1], size[0]
        # for i in range(boxes_filt.size(0)):
        #     boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        #     boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        #     boxes_filt[i][2:] += boxes_filt[i][:2]
​
        # boxes_filt = boxes_filt #.cpu()
        # transformed_boxes = self.segmenter.transform.apply_boxes_torch(boxes_filt, image.shape[:2])
        # masks, _, _ = self.segmenter.predict_torch(
        #     point_coords = None,
        #     point_labels = None,
        #     boxes = transformed_boxes.to(device),
        #     multimask_output = False,
        # )
​
        # # draw output image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)
​
        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, "grounded_sam_output.jpg"), 
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )
​
        # st()
​
    def get_predictions(self, rgb):
        
        image_pil, image = preprocess_image(rgb)
​
        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            self.grounder, image, self.text_prompt, self.box_threshold, self.text_threshold, device=device
        )
​
        if len(boxes_filt)==0:
            return [], [], []
​
        self.segmenter.set_image(rgb)
​
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
​
        boxes_filt = boxes_filt #.cpu()
        transformed_boxes = self.segmenter.transform.apply_boxes_torch(boxes_filt, rgb.shape[:2])
        masks, _, _ = self.segmenter.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
​
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
​
        # plt.axis('off')
        # plt.savefig(
        #     f'data/images/openset4/opensetseg{self.count}.png', 
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )
        # self.count += 1
​
        to_keep = []
        labels = []
        for phrase in pred_phrases:
            name = phrase.split(' ')[-1][:-6]
            if name in self.parsed_name_to_name.keys():
                labels.append(self.name_to_id[self.parsed_name_to_name[name]])
                to_keep.append(True)
            else:
                to_keep.append(False)
        labels = np.asarray(labels)
        to_keep = np.asarray(to_keep)
​
        scores = np.asarray([float(phrase[-5:-1]) for phrase in pred_phrases])
​
        return masks[to_keep].squeeze(1).cpu().numpy(), labels, scores[to_keep]
​
        
​
​
def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model
​
​
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    # print(caption)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]
​
    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]
​
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
​
    return boxes_filt, pred_phrases
​
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
​
​
def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)
​
​
def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background
​
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
​
    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
​
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
​
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
​
def preprocess_image(image):
    image_pil = Image.fromarray(image)
​
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
​
​
if __name__ == "__main__":
    opensetsegmenter = OpenSetSegmenter()