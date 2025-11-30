## `cifar_common.py` 개요
Microdegree 실험 전체에서 반복 사용되는 데이터 로더/모델/학습 루틴을 모아둔 모듈입니다. 노트북에서 동일한 환경을 유지하기 위해 아래 함수들을 직접 import 합니다.

### 공통 설정
- `device`: CUDA 사용 가능 여부에 따라 자동 선택.
- `set_seed(seed=42)`: Python `random`, NumPy, PyTorch(CUDA 포함)까지 모두 동일 시드를 적용해 재현성을 확보합니다.
- `ensure_dir(path)`: 결과 저장 폴더가 없을 때 생성.

### CIFAR-10 DataLoader 생성 (`create_cifar10_loaders`)
- 인자: `root='./data'`, `batch_size=128`, `data_ratio=1.0`, `for_vit=False`.
- CNN과 ViT에 맞는 전처리 파이프라인을 미리 정의(`train_transform_cnn`, `train_transform_vit`)하고 `for_vit` 플래그에 따라 선택합니다.
- `data_ratio` < 1.0이면 train 데이터 일부만 Subset으로 골라 데이터 효율성 실험을 지원합니다.

### 모델 생성 함수
- `create_resnet18_cifar(num_classes=10, pretrained=False)`
  - torchvision ResNet-18을 불러와 FC 층을 CIFAR-10 클래스 수에 맞게 교체합니다.
  - `pretrained=True`일 때 ImageNet 가중치를 사용합니다.
- `create_vit_model(model_name='vit_tiny_patch16_224', num_classes=10, pretrained=True)`
  - `timm` 기반 Vision Transformer를 생성하며, 원하는 모델 이름과 클래스 수를 지정할 수 있습니다.

### 학습/평가 루틴
- `train_one_epoch(model, train_loader, criterion, optimizer, device)`
  - 한 epoch 동안 forward/backward/optimizer step을 수행하고 평균 loss, accuracy를 반환합니다.
  - `non_blocking=True`로 GPU 전송을 최적화했습니다.
- `evaluate(model, data_loader, criterion, device)`
  - gradient 계산을 끈 상태에서 Loss/Accuracy만 측정하며 검증·테스트 공용으로 사용합니다.

### 사용 가이드
1. 노트북 상단에서 `from cifar_common import (...)` 형태로 필요한 함수만 import 합니다.
2. 실험 시작 전에 반드시 `set_seed()`를 호출해 랜덤성을 고정합니다.
3. 새로운 결과를 저장할 때는 `ensure_dir()` 또는 Python `os.makedirs(..., exist_ok=True)`를 활용해 폴더를 만든 뒤 CSV/모델을 기록합니다.
4. CNN/ViT 등 모델별 학습 하이퍼파라미터는 노트북에서 정의하지만, 데이터 전처리/로더/기본 학습 루프는 이 모듈을 재사용해 코드 중복을 줄입니다.
