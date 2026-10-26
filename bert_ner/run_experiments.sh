#!/bin/bash
#SBATCH --job-name=pfas_ner
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=pfas_ner_%j.out
# Full grid: 3 encoders x 4 fine-tuning methods x 5 seeds -> runs/, then summary.
# Usage (from bert_ner/): bash run_experiments.sh    (or: sbatch run_experiments.sh)
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export TOKENIZERS_PARALLELISM=false
PY=${PYTHON:-python}
model_id() {
  case $1 in
    bert)      echo bert-base-cased ;;
    biomed)    echo microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext ;;
    pubmedabs) echo microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract ;;
  esac
}
for SEED in 42 43 44 45 46; do
  for K in bert biomed pubmedabs; do
    M=$(model_id $K)
    $PY train_ner.py --model $M --name ${K}_full --method full --lr 5e-5 --epochs 20 --seed $SEED
    $PY train_ner.py --model $M --name ${K}_lora --method lora --lr 5e-4 --epochs 40 --seed $SEED
    for A in pfeiffer houlsby; do
      $PY train_ner.py --model $M --name ${K}_${A} --method $A --lr 5e-4 --epochs 40 --seed $SEED
    done
  done
done
$PY aggregate.py      # -> results/results.txt
