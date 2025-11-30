## 목표
- CIFAR-10 깨끗한(train/test) 데이터를 동일 조건에서 CNN(ResNet-18)과 ViT(vit_tiny_patch16_224)로 학습해 **baseline 성능 곡선**을 확보한다.
- epoch 수(5/10/20)에 따른 학습/검증 정확도 흐름과 최종 정확도를 저장해 이후 실험(데이터 효율성, robust) 비교의 기준으로 사용한다.

## 실험 구성
1. 공통 전처리/모델 생성 함수를 `cifar_common.py`에서 불러온다.
2. `run_baseline_experiment()`이 모델 유형에 맞는 DataLoader/optimizer를 구성하고, epoch마다 train/test 지표를 기록한다.
3. epoch 리스트 `[5, 10, 20]`에 대해 CNN, ViT을 각각 학습하여 `results/baseline/<model>/`에 CSV(loss/acc 기록)와 학습된 weight(`.pth`)를 저장한다.
4. 가장 긴 epoch의 학습/검증 곡선을 한 그래프로 그려 **수렴 속도 차이**를 직관적으로 확인한다.
5. epoch 별 최종 정확도를 텍스트로 출력해 보고서 작성 시 그대로 인용할 수 있도록 한다.

## 산출물/활용 포인트
- `results/baseline/cnn|vit/` 하위 CSV/모델을 이후 데이터 효율성, CIFAR-10-C 평가에서 불러 활용.
- baseline 정확도 표/그래프를 Microdegree 보고서의 “실험 환경” 섹션에 삽입하면, 이후 실험과의 상대 비교가 명확해진다.
