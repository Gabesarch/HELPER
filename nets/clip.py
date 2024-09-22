from PIL import Image
import numpy as np
import ipdb
st = ipdb.set_trace
import torch
# import clip
from arguments import args

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIP:
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel

        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
        self.preprocess = CLIPProcessor.from_pretrained(args.clip_model)

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    @torch.no_grad()
    def score(self, image=None, texts=None):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.preprocess(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        return probs

    @torch.no_grad()
    def score_images(self, image_query=None, images=None):

        input_query = self.preprocess(text=None, images=image_query, return_tensors="pt", padding=True).to(device)
        image_features_query = self.model.get_image_features(**input_query)

        if isinstance(images, torch.Tensor):
            image_features = images.to(device)
        else:
            inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
            image_features= self.model.get_image_features(**inputs)

        probs = self.cos_sim(image_features_query, image_features)

        return probs

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.preprocess(text=None, images=images, return_tensors="pt", padding=True).to(device)
        image_features = self.model.get_image_features(**inputs)

        return image_features

class ALIGN:
    def __init__(self):
        from transformers import AlignProcessor, AlignModel

        self.preprocess = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()

    @torch.no_grad()
    def score(self, image, texts):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.preprocess(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        return probs

        