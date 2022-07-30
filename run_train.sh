# TODO 加大epoch 5到8 调整warm_step 42到68 去除缩放因子 learning_rate 6e-5 to 3e-05

epoch=8
iter_per_epoch=85
warmup_steps=68
learning_rate=3e-05
hard_negatives=4
batch_size=1024
train_data_path=retrieval_train_data
pretrained_model_cfg=/home/yy/pretrainModel/chinese-macbert-base
output_dir=./outputs/macbert_model_ckp
emb_out_file=./outputs/macbert_context_emb
CUDA_VISIBLE_DEVICES=3


  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_dense_encoder.py \
     --max_grad_norm 2.0 \
     --encoder_model_type hf_bert \
     --pretrained_model_cfg ${pretrained_model_cfg} \
     --seed 12345 \
     --q_sequence_length 32 \
     --p_sequence_length 384 \
     --warmup_steps ${warmup_steps} \
     --batch_size ${batch_size} \
     --do_lower_case \
     --train_file ./data/dureader_data/${train_data_path}/dual_train.json \
     --dev_file ./data/dureader_data/retrieval_train_data_from_baseline/dev_with_hn.json \
     --output_dir ./${output_dir} \
     --learning_rate ${learning_rate} \
     --num_train_epochs ${epoch} \
     --dev_batch_size 16 \
     --val_av_rank_start_epoch $[epoch-1] \
     --grad_cache \
     --global_loss_buf_sz 2097152 \
     --val_av_rank_max_qs 1000 \
     --q_chunk_size 512 \
     --ctx_chunk_size 64 \
     --fp16 \
     --log_batch_step 20 \
     --hard_negatives ${hard_negatives} \
     --other_negatives 0 \

