"""
Name : prepare_hard_negatives.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/5/3 10:19
Desc: 接收重排器预测的训练集top-200结果，选择得分低于 0.1 的最高检索段落作为 hard negatives
"""
import json
import pandas as pd

retriever = '0.589'


"""加载top200数据
        单条格式：
        {'q_text': '',
       'q_id': '',
       'top_n': [(doc_id, doc_score), (...)]}
        """
with open(f'../{retriever}/train_data_top200_with_scores.json', 'r', encoding='utf-8') as f:
    train_data_top_n_with_scores = json.load(f)


'''
加载cleaned_passages
'''
passage_dict = {}
data = pd.read_csv('./dureader_data/cleaned_passages.tsv', delimiter='\t', chunksize=2024167)
for sub_data in data:
    for pid, text in zip(sub_data['id'], sub_data['text']):
        passage_dict[pid] = text
print(f'cleaned_passages numbers: {len(passage_dict)}')


'''
选择{retriever}模型检索回的得分低于 0.1 的最高检索段落作为hard_negatives(只取前12个)作为新的检索训练集
train.json格式：
[{'question_id': '...', 'question': '...', 'answer_paragraphs': [{'paragraph_id': '...', 'paragraph_text': '...'}, ...]}, ...]
融合./{retriever}/train_data_top200_with_scores.json 和 train.json 成新的 dual_train.json 文件：
[  
  {  
   "question": "....",  
   "answers": ["...", "...", "..."],  
   "positive_ctxs": [{  
      "title": "...",  
      "text": "...."  
   }],  
   "negative_ctxs": ["..."],  
   "hard_negative_ctxs": [{  
      "title": "...",  
      "text": "...."  
   }]  
  },  
  ...  
]  
'''
output_data_path = f'./dureader_data/retrieval_train_data_from_{retriever}/dual_train.json'
train_path = './dureader_data/dureader_retrieval-data/train.json'

train_file = open(train_path, 'r', encoding='utf-8')

dual_train = []
train_file_list = [json.loads(i) for i in train_file.readlines()]

for item1, item2 in zip(train_file_list, train_data_top_n_with_scores):
    assert item1["question_id"] == item2["q_id"]
    temp = {
        "question": item1["question"],
        "answers": [],
        "positive_ctxs": [],
        "negative_ctxs": [],
        "hard_negative_ctxs": [],
    }
    answer_paragraphs = item1["answer_paragraphs"]
    answer_paragraphs_ids = [i["paragraph_id"] for i in answer_paragraphs]
    positive_ctxs = [{"title": "",
                      "text": i["paragraph_text"]} for i in answer_paragraphs]
    hard_negative_ctxs = []
    # 选择得分低于 0.1 的最高检索段落（取前12条）
    for res in item2["top_n"]:
        if len(hard_negative_ctxs) >= 12:
            break
        if res[1] < 0.1 and res[0] not in answer_paragraphs_ids:
            hard_negative_ctxs.append({
                "title": "",
                "text": passage_dict[res[0]]
            })
    if len(hard_negative_ctxs) < 12:
        print(f"问题 {item2['q_id']} 的 hard_negatives 少于 12 条！")

    temp["hard_negative_ctxs"] = hard_negative_ctxs
    temp["positive_ctxs"] = positive_ctxs
    dual_train.append(temp)

print(f"dual_train len：{len(dual_train)}")
with open(output_data_path, 'w', encoding='utf-8') as f:
    json.dump(dual_train, f, ensure_ascii=False, indent=4)
