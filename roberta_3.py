python "train_roberta3_final.py" --mode train --pre_trained_model_name_or_path roberta-base --train_path "data/train.csv" --val_path "data/val.csv" --test_path "data/test.csv" --lr 0.0001 --max_len 50 --max_epochs 30 --batch_size 128 --model_saving_path "roberta3"

#Execute the command
try:
    python train_roberta3_final.py --mode=predict --pre_trained_model_name_or_path="roberta3/roberta-base" --predict_data_path=data/test.csv --test_saving_path=final_results.csv
    # Command executed successfully
    print("Command executed successfully.")
except Exception as e:
    # Exception occurred
    print("An error occurred:", e)