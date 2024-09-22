from PIL import Image
import numpy as np
import ipdb
st = ipdb.set_trace
import torch
# import clip


device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIP:
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def score(self, image, texts):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.preprocess(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        return probs

class ALIGN:
    def __init__(self):
        from transformers import AlignProcessor, AlignModel

        self.preprocess = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)

    @torch.no_grad()
    def score(self, image, texts):

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.preprocess(text=texts, images=image, return_tensors="pt", padding=True).to(device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        return probs

        