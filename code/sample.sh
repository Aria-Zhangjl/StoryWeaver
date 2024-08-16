# pre-trained sd dir
export MODEL_NAME=''
# trained ckpt dir
export CKPT_RANK=''
# story txt path
export STORY_TXT_DIR=''
# time-aware scale
export SCALE=''
# sample dir
export SAMPLE_DIR=''
# max number of character in per scene
export MAX_CHAR=''
# txt dir for appearance description for character
export CHAR_CAP=''
accelerate launch sample.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_dir=$STORY_TXT_DIR \
  --resolution=512 \
  --output_dir=$SAMPLE_DIR \
  --seed=42 \
  --mask_maker_dir=$MASK_MAKER \
  --max_char=$MAX_CHAR \
  --caption_dir=$CHAR_CAP \
  --ckpt_rank=$CKPT_RANK \
  --seed_num=1 \
  --scale=$SCALE \