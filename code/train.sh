export HF_ENDPOINT=https://hf-mirror.com
export MODEL_NAME='runwayml/stable-diffusion-v1-5'  # pre-trained sd dir
export DATA_DIR='../TBC_Bench/train-folder/Pororo_the_Little_Penguin' # training data path, taking Pororo as an example
export TEST_DATA_DIR='../TBC_Bench/eval-plot/eval_train/each/Pororo_the_Little_Penguin' # eval data path, taking Pororo as an example
export OUTPUT_DIR='output/Pororo' # output dir
export MAX_CHAR=2 # max number of the characters in per story scene
export DATA_NAME='Pororo' #taking Pororo as an example
CUDA_VISIBLE_DEVICES=4 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_dir=$DATA_DIR \
  --test_dataset_dir=$TEST_DATA_DIR \
  --character_dir=$DATA_DIR'/character' \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --checkpointing_steps=200 \
  --max_train_steps=20000 \
  --learning_rate=7e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --seed=42 \
  --max_char=$MAX_CHAR \
  --data_name=$DATA_NAME \
