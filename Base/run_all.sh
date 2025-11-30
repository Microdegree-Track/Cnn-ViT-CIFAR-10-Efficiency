set -euo pipefail
NOTEBOOKS=(
  "01_CIFAR10_CNN_ViT_CleanBase.ipynb"
  "02_CIFAR10_DataEfficiency_CNN_ViT.ipynb"
  "03_CIFAR10_Robustness_CNN_ViT.ipynb"
)
for nb in "${NOTEBOOKS[@]}"; do
  echo "[run_all] Executing $nb"
  jupyter nbconvert --to notebook --execute "$nb" --inplace
done
