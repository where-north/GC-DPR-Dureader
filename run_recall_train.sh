
epoch=8
iter_per_epoch=85
n_docs=100
checkpoint_dir=0.589

# 对训练集进行召回（Top-200）
python dense_retriever.py \
   --dureader_test \
   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/cleaned_passages.tsv \
   --qa_file ./data/dureader_data/extra_train_data/extra_train_data.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ./${checkpoint_dir}/macbert_encoded_extra_train_q.pkl \
   --encode_q_and_save \


python dense_retriever.py \
   --dureader_test \
   --model_file ./${checkpoint_dir}/macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/cleaned_passages.tsv \
   --qa_file ./data/dureader_data/extra_train_data/extra_train_data.json \
   --encoded_ctx_file ./${checkpoint_dir}/macbert_context_emb/\*.pkl \
   --out_file ./extra_train_res.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ./${checkpoint_dir}/macbert_encoded_extra_train_q.pkl \

