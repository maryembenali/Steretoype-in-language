#!/bin/bash
#SBATCH --time=0-00:10:00
#SBATCH --account=def-atrabels-ab
#SBATCH --mem=64000M
#SBATCH --gpus-per-node=p100:1
#SBATCH --cpus-per-task=16

nvidia-smi

module load python/3.10
module load cuda cudnn

virtualenv --no-download ~/envs/mariyam_roberta
source ~/envs/mariyam_roberta/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r ~/projects/def-atrabels-ab/ubaida/Steretoype-in-language/requirement.txt
python ~/projects/def-atrabels-ab/ubaida/Steretoype-in-language/train_roberta3_final.py --mode train --pre_trained_model_name_or_path roberta-base --train_path "~/projects/def-atrabels-ab/ubaida/Steretoype-in-language/data/train.csv" --val_path "~/projects/def-atrabels-ab/ubaida/Steretoype-in-language/data/val.csv" --test_path "~/projects/def-atrabels-ab/ubaida/Steretoype-in-language/data/test.csv" --lr 0.0001 --max_len 50 --max_epochs 30 --batch_size 128 --model_saving_path "roberta3"
# python ~/projects/def-atrabels-ab/ubaida/Steretoype-in-language/train_roberta3_final.py --mode=predict --pre_trained_model_name_or_path="roberta3/roberta-base" --predict_data_path=data/test.csv --test_saving_path="final_results.csv"
