
epoch=8
iter_per_epoch=85
n_docs=50


python dense_retriever.py \
   --dureader_test \
   --model_file ./macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/passages.tsv \
   --qa_file ./data/dureader_data/dureader-retrieval-test1/test1.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ./macbert_encoded_q.pkl \
   --encode_q_and_save \


python dense_retriever.py \
   --dureader_test \
   --model_file ./macbert_model_ckp/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
   --ctx_file  ./data/dureader_data/passages.tsv \
   --qa_file ./data/dureader_data/dureader-retrieval-test1/test1.json \
   --encoded_ctx_file './macbert_context_emb/*.pkl' \
   --out_file ./test_res.json \
   --q_sequence_length 32 \
   --p_sequence_length 384 \
   --n_docs ${n_docs} \
   --validation_workers 32 \
   --q_encoding_path ./macbert_encoded_q.pkl \
