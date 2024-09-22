import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
from torch.optim import SGD, lr_scheduler
import numpy as np
from pycocotools.cocoeval import COCOeval
import os

# Custom dataset class for Coco
class CocoDataset(CocoDetection):
    def __getitem__(self, index):
        img, target = super(CocoDataset, self).__getitem__(index)
        img = F.to_tensor(img)
        return img, target

# Load the COCO dataset
dataset = CocoDataset(root='path/to/coco/images', annFile='path/to/coco/annotations/instances_train2017.json', transform=T.ToTensor())
dataset_test = CocoDataset(root='path/to/coco/images', annFile='path/to/coco/annotations/instances_val2017.json', transform=T.ToTensor())

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load a pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 91  # COCO has 80 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set up the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    lr_scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    # Validation
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            val_losses = sum(loss for loss in loss_dict.values())
            print(f"Validation Loss: {val_losses}")

# Function to calculate mAP
def evaluate_model(model, data_loader, device):
    model.eval()
    coco = data_loader.dataset.coco
    coco_evaluator = COCOeval(coco, coco, iouType='bbox')
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                results.extend([
                    {
                        "image_id": image_id,
                        "category_id": labels[i],
                        "bbox": box.tolist(),
                        "score": scores[i],
                    }
                    for i, box in enumerate(boxes)
                ])
    coco_evaluator.cocoDt = coco_evaluator.coco.loadRes(results)
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator.stats

# Evaluate the model performance on the validation set
metrics = evaluate_model(model, val_loader, device)
print(f"mAP: {metrics[0]}")
