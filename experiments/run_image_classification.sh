#! /usr/bin/bash
echo "Starting image classification experiments..."

# experiments/run_image_classification.sh
#  microsoft/resnet-50 \
# google/mobilenet_v2_1.0_224
# nvidia/mit-b2
#google/vit-base-patch16-224-in21k
#google/vit-base-patch16-224
# --resume_from_checkpoint /home/qzhao9/code/9800/gradproj/experiments/runs/hf_img_classifier/mit-b2/best \
# define a list of model names to iterate over
models=("google/mobilenet_v2_1.0_224"
         "microsoft/resnet-50"
         "nvidia/mit-b2"
         "google/vit-base-patch16-224-in21k"
         "google/vit-base-patch16-224")
for model_name in "${models[@]}"; do
echo "Running training for model: $model_name"

  accelerate launch run_image_classification_no_trainer.py \
  --model_name_or_path $model_name \
  --train_dir /home/qzhao9/datasets/LIMUC/train \
  --validation_dir /home/qzhao9/datasets/LIMUC/test \
  --output_dir runs/hf_img_classifier/$(echo $model_name | tr '/' '_')/ \
  --checkpointing_steps best \
  --seed 27603 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 8 \
  --num_train_epochs 100 \
  --report_to tensorboard \
  --with_tracking \
  --learning_rate 1e-5 \
  --lr_scheduler_type constant_with_warmup \
  --num_warmup_steps 500 \
  --weight_decay 0.01 \
  --num_cycles 10 \
  --ignore_mismatched_sizes \
  --early_stopping_patience 20 \
  --image_size 224 
done

# accelerate launch run_image_classification_no_trainer.py \
# --model_name_or_path google/mobilenet_v2_1.0_224 \
# --resume_from_checkpoint /home/qzhao9/code/9800/gradproj/experiments/runs/hf_img_classifier/mobilenet_v2_1.0_224/best \
# --train_dir /home/qzhao9/datasets/LIMUC/train \
# --validation_dir /home/qzhao9/datasets/LIMUC/test \
# --output_dir runs/hf_img_classifier/mobilenet_v2_1.0_224/ \
# --checkpointing_steps best \
# --seed 27603 \
# --gradient_accumulation_steps 4 \
# --per_device_train_batch_size 8 \
# --num_train_epochs 100 \
# --report_to tensorboard \
# --with_tracking \
# --learning_rate 1e-5 \
# --lr_scheduler_type constant_with_warmup \
# --num_warmup_steps 500 \
# --weight_decay 0.01 \
# --num_cycles 10 \
# --ignore_mismatched_sizes \
# --early_stopping_patience 20 \
# --image_size 224 \