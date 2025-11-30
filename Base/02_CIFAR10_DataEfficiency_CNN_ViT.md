## 목표
- CIFAR-10 학습 데이터의 사용 비율(`data_ratio`: 1.0/0.5/0.2/0.1)을 줄여가며 **데이터 효율성** 관점에서 CNN과 ViT의 성능 저하 패턴을 비교한다.

## 실험 구성
1. `run_experiment()`이 주어진 `data_ratio`에 맞게 train set을 부분 샘플링하고, 동일 epoch(20) 동안 학습/평가 지표를 기록한다.
2. CNN은 ResNet-18(SGD), ViT는 vit_tiny_patch16_224(AdamW) 설정으로 baseline과 동일한 하이퍼파라미터를 유지한다.
3. ratio 별 history를 CSV로 저장하고, best test accuracy만 추려 `df_summary` 테이블을 구성한다.
4. `Train data ratio vs. best test accuracy` 라인을 통해 데이터가 부족할수록 어떤 모델이 더 견디는지 시각화한다.
5. ratio 별 정량 비교를 텍스트로 출력해 보고서의 “Data efficiency 결과”에 바로 활용한다.

## 산출물/활용 포인트
- `results/data_efficiency/<model>/cnn_ratio_*.csv` : ratio마다 학습 곡선 기록.
- 그래프/표를 사용해 RQ1(데이터가 줄어들 때 성능 변화) 답변에 직접 인용 가능.
