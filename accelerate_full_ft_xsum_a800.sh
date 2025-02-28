DATA_NAME=xsum
WEIGHT_DECAY=0.01
OFFLOAD_RATIO=0.0
LR=1e-6
BATCH=8
EPOCHS=1
SEED=43
output_dir=outputs/${DATA_NAME}/opt-6.7b/full_ft_lr${LR}_e${EPOCHS}_bz${BATCH}_wd${WEIGHT_DECAY}_seed${SEED}
mkdir -p ${output_dir}
log=${output_dir}/training.log

CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch run_full_ft_summarization_ds.py \
  --model_name_or_path /home/tangqiansong/program_2/model/opt-6.7b \
  --dataset_name ${DATA_NAME} \
  --max_source_length 256 \
  --max_target_length 64 \
  --val_max_target_length 320 \
  --preprocessing_num_workers 16 \
  --pad_to_max_length \
  --weight_decay ${WEIGHT_DECAY} \
  --per_device_train_batch_size ${BATCH} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate ${LR} \
  --warmup_ratio 0.06 \
  --num_train_epochs ${EPOCHS} \
  --seed ${SEED} \
  --offload_ratio ${OFFLOAD_RATIO} \
  --output_dir ${output_dir} > ${log} 2>&1
