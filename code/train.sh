# pre-trained sd dir
export MODEL_NAME=''
# training data path
export DATA_DIR=''
# eval data path
export TEST_DATA_DIR=''
# output dir
export OUTPUT_DIR=''
# max number of character in per scene
export MAX_CHAR=''

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_dir=$DATA_DIR \
  --test_dataset_dir=$TEST_DATA_DIR \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --checkpointing_steps=200 \
  --max_train_steps=2000 \
  --learning_rate=7e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --seed=42 \
  --max_char=$MAX_CHAR \