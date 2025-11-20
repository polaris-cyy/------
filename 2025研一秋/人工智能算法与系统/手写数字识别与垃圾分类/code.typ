#import "template.typ": *

#let code = (

augmentation: ```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(int(224*1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```,

construction: ```python
import torch.nn as nn
from torchvision import models

model = models.mobilenet_v3_small(pretrained=True)
in_features = model.classifier[0].in_features
out_features = model.classifier[0].out_features
num_classes = 26

model.classifier = nn.Sequential(
    nn.Linear(in_features, out_features),
    nn.Dropout(p=0.2),
    nn.Linear(out_features, num_classes),
)
model.to(device)
```,

freeze: ```python
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

```,

validation-and-save: ```python
@torch.no_grad()
def evaluate(val_loader, model, device):
    model.eval()
    val_corrects = 0
    val_total = 0
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        val_corrects += torch.sum(preds == labels).item()
        val_total += imgs.size(0)
    return val_corrects / val_total

```,

predict-api: ```python
from PIL import Image
import numpy as np

inverted = {0:'Plastic Bottle',1:'Hats',2:'Newspaper',3:'Cans',4:'Glassware',5:'Glass Bottle',6:'Cardboard',7:'Basketball',8:'Paper',9:'Metalware',10:'Disposable Chopsticks',11:'Lighter',12:'Broom',13:'Old Mirror',14:'Toothbrush',15:'Dirty Cloth',16:'Seashell',17:'Ceramic Bowl',18:'Paint bucket',19:'Battery',20:'Fluorescent lamp',21:'Tablet capsules',22:'Orange Peel',23:'Vegetable Leaf',24:'Eggshell',25:'Banana Peel'}

def _load_model_cpu(model_path="results/best_model.pth", num_classes=26):
    global _loaded_model, _class_names
    if _loaded_model is not None:
        return _loaded_model, _class_names
    ckpt = torch.load(model_path, map_location="cpu")
    model = models.mobilenet_v3_small(pretrained=False)
    in_features = model.classifier[0].in_features
    out_features = model.classifier[0].out_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Dropout(p=0.2),
        nn.Linear(out_features, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _loaded_model = model
    _class_names = ckpt.get("class_names", None)
    return model, _class_names

def predict(image_rgb):
    model, class_names = _load_model_cpu()
    if isinstance(image_rgb, np.ndarray):
        img = Image.fromarray(image_rgb.astype('uint8'), mode='RGB')
    else:
        img = image_rgb
    x = _preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
    return inverted[pred_idx]

```,

train: ```python
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    # 计算训练损失和准确率
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # 验证阶段
    val_acc = evaluate(val_loader, model, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": train_ds.classes,
            "epoch": epoch,
            "val_acc": val_acc
        }, "results/best_model.pth")
```,

) // End of namespace code