
epoch=8
iter_per_epoch=85
n_docs=50
checkpoint_dir=0.566

# 纯验证
 CUDA_VISIBLE_DEVICES=0 python train_dense_encoder.py \
  --q_sequence_length 32 \
  --p_sequence_length 384 \
  --dev_file ./data/dureader_data/retrieval_train_data_from_baseline/dual_dev.json \
  --dev_batch_size 16 \
  --val_av_rank_start_epoch 2 \
  --val_av_rank_max_qs 1000 \
  --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
  --fp16


# 对验证集进行召回（Top-50）
#python dense_retriever.py \
#   --dureader_test \
#   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
#   --ctx_file  ./data/dureader_data/passages.tsv \
#   --qa_file ./data/dureader_data/dureader_retrieval-data/dev.json \
#   --q_sequence_length 32 \
#   --p_sequence_length 384 \
#   --n_docs ${n_docs} \
#   --validation_workers 32 \
#   --q_encoding_path ./macbert_encoded_dev_q.pkl \
#   --encode_q_and_save \
#
#
#python dense_retriever.py \
#   --dureader_test \
#   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
#   --ctx_file  ./data/dureader_data/passages.tsv \
#   --qa_file ./data/dureader_data/dureader_retrieval-data/dev.json \
#   --encoded_ctx_file './${checkpoint_dir}/macbert_context_emb/*.pkl' \
#   --out_file_or_path ./macbert_dev_res.json \
#   --q_sequence_length 32 \
#   --p_sequence_length 384 \
#   --n_docs ${n_docs} \
#   --validation_workers 32 \
#   --q_encoding_path ./macbert_encoded_dev_q.pkl \


