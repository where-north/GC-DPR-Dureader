
epoch=8
iter_per_epoch=85
n_docs=50
checkpoint_dir=0.672

## 对训练集进行召回（Top-200）
#CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
#   --dureader_test \
#   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
#   --ctx_file  ./data/dureader_data/cleaned_passages.tsv \
#   --qa_file ./data/dureader_data/dureader_retrieval-data/train.json \
#   --q_sequence_length 32 \
#   --p_sequence_length 384 \
#   --n_docs ${n_docs} \
#   --validation_workers 32 \
#   --q_encoding_path ./${checkpoint_dir}/macbert_encoded_train_q.pkl \
#   --encode_q_and_save \
#
#
#CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
#   --dureader_test \
#   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
#   --ctx_file  ./data/dureader_data/cleaned_passages.tsv \
#   --qa_file ./data/dureader_data/dureader_retrieval-data/train.json \
#   --encoded_ctx_file ./${checkpoint_dir}/macbert_context_emb/\*.pkl \
#   --out_file ./train_res.json \
#   --q_sequence_length 32 \
#   --p_sequence_length 384 \
#   --n_docs ${n_docs} \
#   --validation_workers 32 \
#   --q_encoding_path ./${checkpoint_dir}/macbert_encoded_train_q.pkl \



# 对验证集进行召回（Top-50）
CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
   --dureader_test \
   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/cleaned_passages.tsv \
   --qa_file ./data/dureader_data/dureader_retrieval-data/dev.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ./${checkpoint_dir}/macbert_encoded_dev_q.pkl \
   --encode_q_and_save \


CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
   --dureader_test \
   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/cleaned_passages.tsv \
   --qa_file ./data/dureader_data/dureader_retrieval-data/dev.json \
   --encoded_ctx_file ./${checkpoint_dir}/macbert_context_emb/\*.pkl \
   --out_file ./dev_res.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ./${checkpoint_dir}/macbert_encoded_dev_q.pkl \

