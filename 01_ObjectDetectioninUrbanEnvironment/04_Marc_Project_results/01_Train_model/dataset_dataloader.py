import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import os
import numpy as np
import pandas as pd
from PIL import Image
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from engine import train_one_epoch, evaluate
import utils
import transforms as T

class CustomDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotation = annotation
        with open(annotation) as f:
            self.coco = json.load(f)

        self.images = list(sorted(self.coco['images'], key=lambda x: x['id']))
        self.annotations = list(sorted(self.coco['annotations'], key=lambda x: x['image_id']))

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Get the annotations
        ann_ids = [i for i, x in enumerate(self.annotations) if x['image_id'] == img_info['id']]
        anns = [self.annotations[i] for i in ann_ids]

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin+w, ymin+h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Use your own dataset path and annotations path
dataset = CustomDataset(root="path/to/images", annotation="path/to/annotations.json", transforms=get_transform(train=True))
dataset_test = CustomDataset(root="path/to/images", annotation="path/to/annotations.json", transforms=get_transform(train=False))

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
