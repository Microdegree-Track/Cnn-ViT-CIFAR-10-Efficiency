# CNN vs Vision Transformer: CIFAR-10 Clean & Corrupted 실험 

이 레포는 Microdegree (CNN vs Vision Transformer 비교)의 실험 코드를 정리한 것입니다. 세 개의 Jupyter 노트북과 공통 모듈(`cifar_common.py`)을 통해 데이터 효율성과 강인성(Robustness) 관점에서 두 모델을 반복적으로 비교할 수 있도록 구성했습니다.

## 연구 질문
1. **RQ1** – 학습 데이터 양이 줄어들 때 두 모델의 정확도/손실은 어떻게 변하는가?
2. **RQ2** – 깨끗한 테스트셋 대비 CIFAR-10-C corruption에서 성능 저하 양상은 어떻게 다른가?
3. **RQ3** – 간단한 증강/학습 설정 변경이 각 모델의 robustness를 얼마나 개선시키는가?

## 데이터 준비
| 데이터셋 | 설명 | 준비 방법 |
| --- | --- | --- |
| CIFAR-10 | 32×32 컬러 이미지 10클래스 | `torchvision.datasets.CIFAR10`에서 자동 다운로드 (`data/` 하위) |
| CIFAR-10-C | 노이즈·블러·날씨 등 corruption(15종)×severity(1~5) | [Zenodo #2535967](https://zenodo.org/records/2535967)에서 `CIFAR-10-C.tar` 다운로드 → 압축 해제 후 `data/CIFAR-10-C-1/` 위치에 복사 |

> `03_CIFAR10_Robustness_CNN_ViT.ipynb`는 `data/CIFAR-10-C-1/labels.npy` 존재 여부를 확인하므로, 위 경로가 정확히 준비돼야 실행이 멈추지 않습니다.

## 환경 세팅 & 실행
1. (선택) Conda/venv 생성 후 활성화
2. `pip install -r requirements.txt`
3. `bash run_all.sh` 또는 노트북을 순서대로 수동 실행

`run_all.sh`는 `jupyter nbconvert --execute --inplace`를 이용해 3개의 노트북을 자동 실행합니다. GPU 없이 CPU만 사용할 경우 실행 시간이 상당히 길 수 있으므로, 필요 시 노트북 안에서 epoch/ratio를 조정하세요.
> **실험 하드웨어 참고**: 모든 수치는 NVIDIA GeForce RTX 4090 (CUDA 12.4, Driver 550.163.01, VRAM 24GB) 4-way 서버에서 GPU 0을 주로 사용해 얻었습니다. 다른 GPU/CPU 환경에서도 실행 가능하지만 학습 시간과 세부 정확도는 달라질 수 있습니다.


## 노트북 구성 요약
| 파일 | 목적 |
| --- | --- |
| `01_CIFAR10_CNN_ViT_CleanBase.ipynb` | CIFAR-10 clean 데이터 기준 CNN vs ViT baseline (epoch 5/10/20) 학습, CSV/가중치 저장 및 학습 곡선 시각화 |
| `02_CIFAR10_DataEfficiency_CNN_ViT.ipynb` | `data_ratio ∈ {1.0, 0.5, 0.2, 0.1}` 에 따라 두 모델의 데이터 효율성 비교, 결과 CSV/그래프 생성 |
| `03_CIFAR10_Robustness_CNN_ViT.ipynb` | baseline 가중치를 로드해 CIFAR-10-C corruption×severity에서 정확도 측정, 요약 테이블 및 시각화 생성 |
| `cifar_common.py/md` | DataLoader/모델 생성/학습 루틴을 모듈화; 문서는 함수별 사용법 설명 |

## 사용한 모델 (CNN/ViT)
- CNN: torchvision ResNet-18
  - 구현: `torchvision.models.resnet18`
  - 출력층: FC를 `num_classes=10`으로 교체
  - 사전학습: `pretrained=False`로 학습 (ImageNet 가중치 미사용)
  - 입력/전처리: 32×32 CIFAR-10 해상도, `RandomCrop(32,pad=4)` + `RandomHorizontalFlip` + CIFAR-10 mean/std 정규화
  - 기본 배치/최적화: batch size 128, SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)
- ViT: timm `vit_tiny_patch16_224`
  - 구현: `timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)`
  - 사전학습: ImageNet-1k 가중치 사용(`pretrained=True`)
  - 입력/전처리: 224 해상도, `Resize(224)` + `RandomResizedCrop(224)`(train) / `CenterCrop(224)`(test) + ImageNet mean/std 정규화
  - 기본 배치/최적화: batch size 64, AdamW(lr=3e-4, weight_decay=0.05)


## 현재 확보된 결과 (RQ1 & RQ2 포함)

### 1) Clean Baseline (CIFAR-10)
`results/baseline/*` CSV에서 추출한 최고 test accuracy는 아래와 같습니다.

| 모델 | Epoch | Best Test Acc |
| --- | --- | --- |
| CNN | 5 | 68.53% |
| CNN | 10 | 73.46% |
| CNN | 20 | 78.65% |
| ViT | 5 | 92.42% |
| ViT | 10 | 93.22% |
| ViT | 20 | 93.96% |

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/7cb5a9ad-4be7-4cf4-b846-eee6c9e21709" />


### 2) 데이터 효율성 (RQ1)
`02_CIFAR10_DataEfficiency_CNN_ViT.ipynb`로 train 데이터 비율을 줄여가며 학습한 결과입니다.

| 모델 | Train ratio | Best Test Acc |
| --- | --- | --- |
| CNN | 1.0 | 78.65% |
| CNN | 0.5 | 73.10% |
| CNN | 0.2 | 62.70% |
| CNN | 0.1 | 51.44% |
| ViT | 1.0 | 93.96% |
| ViT | 0.5 | 92.94% |
| ViT | 0.2 | 91.31% |
| ViT | 0.1 | 89.07% |

- ViT는 데이터가 10% 수준으로 줄어들어도 89% 이상의 정확도를 유지했습니다.
- CNN은 데이터 부족 시 빠르게 성능이 하락해 0.1 비율에서 51%까지 떨어졌습니다.

<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/b5b3899b-4630-4385-896d-6e753ce13b71" />


### 3) Robustness (RQ2)
`03_CIFAR10_Robustness_CNN_ViT.ipynb`는 baseline 가중치를 CIFAR-10-C에 적용해 corruption×severity별 정확도를 측정했습니다.

| Severity | CNN | ViT |
| --- | --- | --- |
| 1 | 77.38% | 88.90% |
| 2 | 74.82% | 82.51% |
| 3 | 71.27% | 72.74% |
| 4 | 68.76% | 67.78% |
| 5 | 62.38% | 58.11% |

- 저강도(severity 1~2)에서는 ViT가 크게 앞서지만, 강도가 올라갈수록 ViT 정확도가 더 빠르게 감소해 severity 4~5에서는 CNN과 비슷하거나 더 낮습니다.
- Corruption 별로 보면 ViT는 밝기·안개·블러에 강하지만 **gaussian_noise/shot_noise**에서는 CNN보다 취약합니다.

| Corruption (severity=3) | CNN | ViT |
| --- | --- | --- |
| brightness | 77.07% | 91.48% |
| fog | 67.28% | 87.98% |
| gaussian_noise | 72.29% | 43.63% |
| motion_blur | 64.70% | 82.34% |
| shot_noise | 74.99% | 58.28% |

<img width="689" height="490" alt="image" src="https://github.com/user-attachments/assets/be2bc177-c137-4d7a-9a0e-c82ecb153464" />
<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/4f539c11-dbf5-4fb4-aaa0-a0c5a8b5b8d3" />


> 전체 평균 정확도: CNN 70.9%, ViT 74.0% (ViT가 전반적으로는 앞서지만 노이즈류에 매우 민감)

## 결론 
- **결론**
  - **Baseline 관찰**: ViT는 동일 epoch budget에서 90% 이상의 test accuracy를 즉시 달성하고, CNN은 epoch을 늘릴수록 꾸준히 개선되는 저비용 baseline 역할을 합니다.
  - **RQ1 결과**: 데이터 비율을 줄이면 CNN 성능이 급격히 저하되는 반면 ViT는 10% 데이터에서도 89% 정확도를 유지해 데이터 효율성 측면에서 명확한 우위를 보였습니다.
  - **RQ2 결과**: ViT는 조명/안개/블러에는 강하지만 노이즈에는 매우 민감해 severity가 높아지면 CNN과 격차가 줄어듭니다. 현실 환경의 노이즈 특성에 따라 모델 선택 전략이 달라질 수 있음을 시사합니다.
