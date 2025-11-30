## 목표
- 깨끗한 CIFAR-10에서 학습한 baseline 모델을 그대로 사용해 **CIFAR-10-C**(noise/blur/weather 등 corruption)에서 CNN vs ViT의 강인성(Robustness) 차이를 정량화한다.

## 실험 구성
1. `CIFAR10C` 커스텀 Dataset을 정의해 corruption 타입·severity(1~5)에 해당하는 1만 장씩을 불러온다.
2. `create_cifar10c_loader()`가 CNN/ViT에 맞는 전처리(32x32 vs 224x224)로 DataLoader를 생성한다.
3. baseline에서 저장한 가중치(`results/baseline/..._epochs_20.pth`)를 로드해 **깨끗한 데이터 재학습 없이** CIFAR-10-C에서만 평가한다.
4. `evaluate_on_cifar10c()`가 corruption × severity 조합을 모두 순회하며 test loss/accuracy를 측정하고, `results/robustness/`에 CSV로 저장한다.
5. 평균 severity 곡선, severity=3 기준 corruption 별 bar plot을 통해 모델별 취약 패턴을 시각화한다.

## 산출물/활용 포인트
- `results/robustness/robustness_cifar10c_epochs20.csv`: CNN/ViT 모두의 세부 지표.
- `robustness_summary.csv`: 모델별 평균 정확도(강인성 수준) 요약.
- 시각화 2종을 RQ2(Noise/Blur 등에서 성능 저하 양상)의 핵심 근거로 활용한다.
- `CIFAR10C_ROOT = data/CIFAR-10-C-1` 경로 존재 여부를 코드에서 확인하므로, 데이터 연결 이슈를 사전에 잡을 수 있다.
