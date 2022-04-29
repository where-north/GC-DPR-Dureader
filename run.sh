
 CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg /home/yy/pretrainModel/chinese-macbert-base \
    --seed 12345 \
    --sequence_length 384 \
    --warmup_steps 4 \
    --batch_size 2048 \
    --do_lower_case \
    --train_file ./data/dureader_data/retrieval_train_data/dual_train.json \
    --dev_file ./data/dureader_data/retrieval_train_data/dev_with_hn.json \
    --output_dir ./macbert_model_ckp \
    --learning_rate 12e-05 \
    --num_train_epochs 3 \
    --dev_batch_size 16 \
    --val_av_rank_start_epoch 2 \
    --grad_cache \
    --global_loss_buf_sz 2097152 \
    --val_av_rank_max_qs 1000 \
    --q_chunk_size 64 \
    --ctx_chunk_size 32 \
    --fp16 \
    --log_batch_step 10 \
    --hard_negatives 4 \
    --other_negatives 0 \



 for n in $(seq 0 3);
 do
     if [ $n -eq 0 ]
         then
 	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages0-0.tsv \
        --out_file ./macbert_context_emb/context_emb_0 \
        --fp16 &
     elif [ $n -eq 1 ]
         then
 	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages0-1.tsv \
        --out_file ./macbert_context_emb/context_emb_1 \
        --fp16 &
     elif [ $n -eq 2 ]
 	then
 	    CUDA_VISIBLE_DEVICES=2 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages0-2.tsv \
        --out_file ./macbert_context_emb/context_emb_2 \
        --fp16 &
     elif [ $n -eq 3 ]
 	then
       CUDA_VISIBLE_DEVICES=3 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages0-3.tsv \
        --out_file ./macbert_context_emb/context_emb_3 \
        --fp16 &
     fi
 done
 wait

 for n in $(seq 0 3);
 do
     if [ $n -eq 0 ]
         then
 	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages1-0.tsv \
        --out_file ./macbert_context_emb/context_emb_4 \
        --fp16 &
     elif [ $n -eq 1 ]
         then
 	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages1-1.tsv \
        --out_file ./macbert_context_emb/context_emb_5 \
        --fp16 &
     elif [ $n -eq 2 ]
 	then
 	    CUDA_VISIBLE_DEVICES=2 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages1-2.tsv \
        --out_file ./macbert_context_emb/context_emb_6 \
        --fp16 &
     elif [ $n -eq 3 ]
 	then
       CUDA_VISIBLE_DEVICES=3 python generate_dense_embeddings.py \
        --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
        --ctx_file ./data/dureader_data/passages1-3.tsv \
        --out_file ./macbert_context_emb/context_emb_7 \
        --fp16 &
     fi
 done
 wait


#python dense_retriever.py \
#   --dureader_test \
#   --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
#   --ctx_file  ./data/dureader_data/passages.tsv \
#   --qa_file ./data/dureader_data/dureader-retrieval-test1/test1.json \
#   --n-docs 50 \
#   --validation_workers 32 \
#   --batch_size 128  \
#   --q_encoding_path ./macbert_encoded_q.pkl \
#   --encode_q_and_save \
#
#
#python dense_retriever.py \
#   --dureader_test \
#   --model_file ./macbert_model_ckp/dpr_biencoder.2.43 \
#   --ctx_file  ./data/dureader_data/passages.tsv \
#   --qa_file ./data/dureader_data/dureader-retrieval-test1/test1.json \
#   --encoded_ctx_file './macbert_context_emb/*.pkl' \
#   --out_file ./macbert_res.json \
#   --n-docs 50 \
#   --validation_workers 32 \
#   --batch_size 128  \
#   --q_encoding_path ./macbert_encoded_q.pkl \


# 纯验证
# CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
#  --sequence_length 384 \
#  --dev_file ./data/dureader_data/retrieval_train_data/dual_dev.json \
#  --dev_batch_size 16 \
#  --val_av_rank_start_epoch 2 \
#  --val_av_rank_max_qs 1000 \
#  --model_file /home/chy/GC-DPR-main/0.453/macbert_model_ckp/dpr_biencoder.2.43 \
#  --fp16


# 对训练集进行召回（Top-50）
#python dense_retriever.py \
#   --dureader_test \
#   --model_file ./0.453/macbert_model_ckp/dpr_biencoder.2.169 \
#   --ctx_file  ./data/dureader_data/passages.tsv \
#   --qa_file ./data/dureader_data/dureader_retrieval-data/train.json \
#   --n-docs 50 \
#   --validation_workers 32 \
#   --batch_size 128  \
#   --q_encoding_path ./macbert_encoded_train_q.pkl \
#   --encode_q_and_save \
#
#
#python dense_retriever.py \
#   --dureader_test \
#   --model_file ./0.453/macbert_model_ckp/dpr_biencoder.2.169 \
#   --ctx_file  ./data/dureader_data/passages.tsv \
#   --qa_file ./data/dureader_data/dureader_retrieval-data/train.json \
#   --encoded_ctx_file './0.453/macbert_context_emb/*.pkl' \
#   --out_file ./macbert_train_res.json \
#   --n-docs 50 \
#   --validation_workers 32 \
#   --batch_size 128  \
#   --q_encoding_path ./macbert_encoded_train_q.pkl \


