# TODO 加大epoch到8 调整warm_step 42到68
 CUDA_VISIBLE_DEVICES=1 python train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg /home/yy/pretrainModel/chinese-macbert-base \
    --seed 12345 \
    --q_sequence_length 32 \
    --p_sequence_length 384 \
    --warmup_steps 68 \
    --batch_size 1024 \
    --do_lower_case \
    --train_file ./data/dureader_data/retrieval_train_data_from_baseline/dual_train.json \
    --dev_file ./data/dureader_data/retrieval_train_data_from_baseline/dev_with_hn.json \
    --output_dir ./macbert_model_ckp \
    --learning_rate 6e-05 \
    --num_train_epochs 8 \
    --dev_batch_size 16 \
    --val_av_rank_start_epoch 4 \
    --grad_cache \
    --global_loss_buf_sz 2097152 \
    --val_av_rank_max_qs 1000 \
    --q_chunk_size 512 \
    --ctx_chunk_size 64 \
    --fp16 \
    --log_batch_step 20 \
    --hard_negatives 4 \
    --other_negatives 0 \


 for n in $(seq 0 3);
 do
     if [ $n -eq 0 ]
         then
 	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages0-0.tsv \
        --out_file ./macbert_context_emb/context_emb_0 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     elif [ $n -eq 1 ]
         then
 	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages0-1.tsv \
        --out_file ./macbert_context_emb/context_emb_1 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     elif [ $n -eq 2 ]
 	then
 	    CUDA_VISIBLE_DEVICES=2 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages0-2.tsv \
        --out_file ./macbert_context_emb/context_emb_2 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     elif [ $n -eq 3 ]
 	then
       CUDA_VISIBLE_DEVICES=3 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages0-3.tsv \
        --out_file ./macbert_context_emb/context_emb_3 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     fi
 done
 wait

 for n in $(seq 0 3);
 do
     if [ $n -eq 0 ]
         then
 	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages1-0.tsv \
        --out_file ./macbert_context_emb/context_emb_4 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     elif [ $n -eq 1 ]
         then
 	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages1-1.tsv \
        --out_file ./macbert_context_emb/context_emb_5 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     elif [ $n -eq 2 ]
 	then
 	    CUDA_VISIBLE_DEVICES=2 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages1-2.tsv \
        --out_file ./macbert_context_emb/context_emb_6 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     elif [ $n -eq 3 ]
 	then
       CUDA_VISIBLE_DEVICES=3 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.7.85 \
        --ctx_file ./data/dureader_data/passages1-3.tsv \
        --out_file ./macbert_context_emb/context_emb_7 \
        --fp16 \
        --q_sequence_length 32 \
        --p_sequence_length 384 &
     fi
 done
 wait




