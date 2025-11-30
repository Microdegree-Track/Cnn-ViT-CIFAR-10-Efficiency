import random
import numpy as np
import torch
import os 
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

try:
    import timm
    HAS_TIMM = True     
except ImportError:
    timm = None
    HAS_TIMM = False   



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CNN/ViT 각각에 맞는 transform을 미리 정의해두고 create_cifar10_loaders에서 선택적으로 사용하도록 구성.
train_transform_cnn = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

test_transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

train_transform_vit = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform_vit = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def create_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    data_ratio: float = 1.0,
    for_vit: bool = False,
):
    """
    CIFAR-10 train/test DataLoader 생성.
    data_ratio: train 데이터 비율 (1.0, 0.5, 0.2, 0.1 등).
    for_vit: ViT용 transform 사용 여부.
    """
    assert 0 < data_ratio <= 1.0

    if for_vit:
        train_transform = train_transform_vit
        test_transform = test_transform_vit
    else:
        train_transform = train_transform_cnn
        test_transform = test_transform_cnn

    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    if data_ratio < 1.0:
        num_total = len(train_dataset)
        num_subset = int(num_total * data_ratio)
        indices = np.random.permutation(num_total)[:num_subset]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader

## CNN Model
def create_resnet18_cifar(num_classes: int = 10, pretrained: bool = False):
    """
    CIFAR-10용 ResNet-18.
    pretrained=True: ImageNet 사전학습 weight 사용.
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

## ViT Model
def create_vit_model(
    model_name: str = "vit_tiny_patch16_224",
    num_classes: int = 10,
    pretrained: bool = True,
):
    """
    timm 기반 Vision Transformer 생성.
    """
    if timm is None:
        raise RuntimeError("timm이 설치되어 있지 않습니다.")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model

## 학습 및 평가

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """한 epoch 학습."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    """검증/테스트 평가."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc