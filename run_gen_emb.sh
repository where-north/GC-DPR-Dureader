
epoch=8
iter_per_epoch=85
output_dir=macbert_model_ckp
emb_out_file=macbert_context_emb

for n in $(seq 0 1);
do
    if [ $n -eq 0 ]
        then
	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages0-0.tsv \
       --out_file ./${emb_out_file}/context_emb_0 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    elif [ $n -eq 1 ]
        then
	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages0-1.tsv \
       --out_file ./${emb_out_file}/context_emb_1 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    fi
done
wait

for n in $(seq 0 1);
do
    if [ $n -eq 0 ]
        then
	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages0-2.tsv \
       --out_file ./${emb_out_file}/context_emb_2 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    elif [ $n -eq 1 ]
        then
	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages0-3.tsv \
       --out_file ./${emb_out_file}/context_emb_3 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    fi
done
wait

for n in $(seq 0 1);
do
    if [ $n -eq 0 ]
        then
	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages1-0.tsv \
       --out_file ./${emb_out_file}/context_emb_4 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    elif [ $n -eq 1 ]
        then
	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages1-1.tsv \
       --out_file ./${emb_out_file}/context_emb_5 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    fi
done
wait

for n in $(seq 0 1);
do
    if [ $n -eq 0 ]
        then
	    CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages1-2.tsv \
       --out_file ./${emb_out_file}/context_emb_6 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    elif [ $n -eq 1 ]
        then
	    CUDA_VISIBLE_DEVICES=1 python generate_dense_embeddings.py \
       --model_file ./${output_dir}/dpr_biencoder.$[epoch-1].${iter_per_epoch} \
       --ctx_file ./data/dureader_data/passages1-3.tsv \
       --out_file ./${emb_out_file}/context_emb_7 \
       --fp16 \
       --q_sequence_length 32 \
       --p_sequence_length 384 &
    fi
done
wait

