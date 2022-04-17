#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l jobfs=10GB
#PBS -q gpuvolta
#PBS -P ik70
#PBS -l walltime=6:00:00
#PBS -l ngpus=1
#PBS -l storage=scratch/ik70
#PBS -l wd
#PBS -M dai.dai@csiro.au
#PBS -m e
#PBS -r y
#PBS -J 0-8


module load intel-mkl/2020.3.304
module load python3/3.9.2
module load cuda/11.2.2
module load cudnn/8.1.1-cuda11
module load openmpi/4.1.0
module load magma/2.6.0
module load fftw3/3.3.8
module load pytorch/1.9.0



cd ../code
data_dir=/scratch/ik70
dataset=mimic50

SEEDs=(52 869 1001)
LENGTHs=(8192 6144 4096)
seed=${SEEDs[$PBS_ARRAY_INDEX/${#LENGTHs[@]}]}
length=${LENGTHs[$PBS_ARRAY_INDEX%${#LENGTHs[@]}]}
output=${dataset}_length_${length}_seed_${seed}
output_dir=${data_dir}/TEMP/0330A_1_${output}_$(date +%F-%H-%M-%S-%N)

if ! test -f "../results/1/test/${output}.json"; then
  python3 train.py \
  --task_name multilabel \
  --dataset_name $dataset \
  --output_metrics_filepath ../results/1/train/${output}.json \
  --model_dir $data_dir/Corpora/RoBERTa/mimic_roberta_base \
  --seed $seed \
  --train_filepath $data_dir/ProcessedData/MIMIC-III/0/50/train.json \
  --dev_filepath $data_dir/ProcessedData/MIMIC-III/0/50/dev.json \
  --output_dir $output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --metric_for_best_model micro_f1 \
  --greater_is_better \
  --max_seq_length $length \
  --segment_length 64 --do_use_stride --do_use_label_wise_attention

  python3 eval.py \
  --task_name multilabel \
  --dataset_name $dataset \
  --output_metrics_filepath ../results/1/test/${output}.json \
  --model_dir $output_dir \
  --test_filepath $data_dir/ProcessedData/MIMIC-III/0/50/test.json \
  --output_dir $output_dir \
  --max_seq_length $length \
  --segment_length 64 --do_use_stride --do_use_label_wise_attention

  rm -r $output_dir
fi
