
epoch=8
iter_per_epoch=85
n_docs=50
model_dir=./outputs
model_ckp=${model_dir}/macbert_model_ckp
context_emb=${model_dir}/macbert_context_emb
q_encoding_path=${model_dir}/test2_macbert_encoded_q.pkl


CUDA_VISIBLE_DEVICES=1 python dense_retriever.py \
   --dureader_test \
   --model_file ${model_ckp}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/passages.tsv \
   --qa_file ./data/dureader_data/dureader-retrieval-test2/test2.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ${q_encoding_path} \
   --encode_q_and_save \


CUDA_VISIBLE_DEVICES=1 python dense_retriever.py \
   --dureader_test \
   --model_file ${model_ckp}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/passages.tsv \
   --qa_file ./data/dureader_data/dureader-retrieval-test2/test2.json \
   --encoded_ctx_file ${context_emb}/\*.pkl \
   --out_file ./test2_res.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ${q_encoding_path} \
