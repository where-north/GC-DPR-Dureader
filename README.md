
### 1. 获取训练Retriever需要的训练集、验证集、语料库集合:  
  
```bash  
python data/preprocess_dureader.py 
```  
  
### 2. Retriever训练  
  
Train on a single 32GB Tesla V100 GPU,
  
```bash  
sh run_train.sh
```  

### 3. 使用训练好的编码语料库中所有段落

```bash  
sh run_gen_emb.sh
```  

### 4. 测试  
  
```bash  
sh run_test.sh
``` 


## Retriever 训练数据格式  
The data format of the Retriever training data is JSON.  
It contains pools of 2 types of negative passages per question, as well as positive passages and some additional information.  

```  
[  
  {  
   "question": "....",  
   "answers": ["...", "...", "..."],  
   "positive_ctxs": [{  
      "title": "...",  
      "text": "...."  
   }],  
   "negative_ctxs": ["..."],  
   "hard_negative_ctxs": ["..."]  
  },  
  ...  
]  
```  
  
  
## Retriever 训练脚本  
```bash  
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
```  
其中，梯度缓存参数：
- `--grad_cache` activates gradient cached training
- `--q_chunk_size` sub-batch size for updating the question encoder, default to 16
- `--ctx_chunk_size` sub-batch size for updating context encoder, default to 8

