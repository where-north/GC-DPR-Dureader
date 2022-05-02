
# 对训练集进行召回（Top-200）
python dense_retriever.py \
   --dureader_test \
   --model_file ./0.589/macbert_model_ckp/dpr_biencoder.7.85 \
   --ctx_file  ./data/dureader_data/passages.tsv \
   --qa_file ./data/dureader_data/dureader_retrieval-data/train.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs 200 \
   --validation_workers 32 \
   --q_encoding_path ./0.589/macbert_encoded_train_q.pkl \
   --encode_q_and_save \


python dense_retriever.py \
   --dureader_test \
   --model_file ./0.589/macbert_model_ckp/dpr_biencoder.7.85 \
   --ctx_file  ./data/dureader_data/passages.tsv \
   --qa_file ./data/dureader_data/dureader_retrieval-data/train.json \
   --encoded_ctx_file './0.589/macbert_context_emb/*.pkl' \
   --out_file_or_path ./0.589/ \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs 200 \
   --validation_workers 32 \
   --q_encoding_path ./0.589/macbert_encoded_train_q.pkl \

