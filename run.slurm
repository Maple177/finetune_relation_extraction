#!/bin/sh
#SBATCH --job-name=finetune_re
#SBATCH --output=run.out
#SBATCH --error=run.out
#SBATCH --nodelist=n55
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=all

#set -x

corpus=chemprot_blurb
num_epochs=60
model_name=pubmedbert
seed=41
lr=1e-05
es=5

printf "run on %s" "$corpus" 

srun python3 main.py --model_name $model_name --task_name $corpus --num_train_epochs $num_epochs --seed $seed  --learning_rate $lr --ensemble_size $es --warmup --do_not_save_all_models 

