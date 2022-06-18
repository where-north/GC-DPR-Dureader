"""
Name : prepare_hard_negatives.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/5/3 10:19
Desc: 接收重排器预测的训练集top-200结果，选择得分低于 0.1 的最高检索段落作为 hard negatives
"""
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm


def get_hard_negatives_from_589():
    retriever = '0.637'

    """加载top n数据
            单条格式：
            {'q_text': '',
           'q_id': '',
           'top_n': [(doc_id, doc_score), (...)]}
            """
    with open(f'../{retriever}/train_data_top50_with_scores.json', 'r', encoding='utf-8') as f:
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
    选择{retriever}模型检索回的得分低于 0.1 的最高检索段落作为hard_negatives(只取前4个)作为新的检索训练集
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
        # 选择得分低于 0.1 的最高检索段落（取前4条）
        scores = [res[1] for res in item2["top_n"]]
        # 归一化scores
        exp_scores = np.exp(scores)
        sum_ = sum(exp_scores)
        for score, res in zip(exp_scores, item2["top_n"]):
            if len(hard_negative_ctxs) >= 4:
                break
            if score / sum_ < 0.1 and res[0] not in answer_paragraphs_ids:
                hard_negative_ctxs.append({
                    "title": "",
                    "text": passage_dict[res[0]]
                })
        if len(hard_negative_ctxs) < 4:
            print(f"问题 {item2['q_id']} 的 hard_negatives 少于 4 条！")

        temp["hard_negative_ctxs"] = hard_negative_ctxs
        temp["positive_ctxs"] = positive_ctxs
        dual_train.append(temp)

    print(f"dual_train len：{len(dual_train)}")
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(dual_train, f, ensure_ascii=False, indent=4)


def get_extra_train_data():
    '''
    extra data
    '''

    """加载top50数据
            单条格式：
            'q_id':{'q_text': '',
           'q_id': '',
           'top_n': [(doc_id, doc_text_doc_score), (...)]}
            """
    # with open(f'../extra_top_n_dict_with_scores.json', 'r', encoding='utf-8') as f:
    #     train_data_top_n_with_scores = json.load(f)

    # output_data_path = f'./dual_extra_train.json'
    # extra_train_path = '/home/chy/GC-DPR-main/data/dureader_data/extra_train_data/extra_train_data.json'
    # train_path = '/home/chy/GC-DPR-main/data/dureader_data/retrieval_train_data_from_0.589/dual_train.json'
    #
    # extra_train_file = open(extra_train_path, 'r', encoding='utf-8')
    # with open(train_path, 'r', encoding='utf-8') as train_file:
    #     dual_train = json.load(train_file)
    # extra_train_file_list = [json.loads(i) for i in extra_train_file.readlines()]
    #
    # for item1 in extra_train_file_list:
    #     q_id = item1["question_id"]
    #     temp = {
    #         "question": item1["question"],
    #         "answers": [],
    #         "positive_ctxs": [],
    #         "negative_ctxs": [],
    #         "hard_negative_ctxs": [],
    #     }
    #     answer_paragraphs = item1["answer_paragraphs"]
    #     answer_paragraphs_ids = [i["paragraph_id"] for i in answer_paragraphs]
    #     positive_ctxs = [{"title": "",
    #                       "text": i["paragraph_text"]} for i in answer_paragraphs]
    #     hard_negative_ctxs = []
    #     # 选择得分低于 0.1 的最高检索段落（取前12条）
    #     for res in train_data_top_n_with_scores[q_id]["top_n"]:
    #         if len(hard_negative_ctxs) >= 12:
    #             break
    #         if res[2] < 0.1 and res[0] not in answer_paragraphs_ids:
    #             hard_negative_ctxs.append({
    #                 "title": "",
    #                 "text": res[1]
    #             })
    #     if len(hard_negative_ctxs) < 12:
    #         print(f"问题 {'q_id'} 的 hard_negatives 少于 12 条！")
    #
    #     temp["hard_negative_ctxs"] = hard_negative_ctxs
    #     temp["positive_ctxs"] = positive_ctxs
    #     dual_train.append(temp)
    #
    # print(f"dual_train len：{len(dual_train)}")
    # with open(output_data_path, 'w', encoding='utf-8') as f:
    #     json.dump(dual_train, f, ensure_ascii=False, indent=4)


def get_hard_negatives_from_672_and_merge_pointwise_model():
    retriever = '0.672'

    def merge_reranker_res():
        bert_wwm_path = f'/home/chy/reranker-main/reranker_model/bert_wwm_pointwise_0.729/'
        nezha_wwm_path = f'/home/chy/reranker-main/reranker_model/nezha_wwm_pointwise_fgm0.01_0.737/'
        macbert_large_path = f'/home/chy/reranker-main/reranker_model/macbert_large_pointwise_0.740/'
        train_data_top_n_with_scores = defaultdict(list)

        for i in range(4):
            bert_top_n_data, nezha_top_n_data, macbert_top_n_data = defaultdict(dict), defaultdict(dict), defaultdict(
                dict)
            print(f'加载{bert_wwm_path}' + f'train_scores_{i}.json')
            with open(bert_wwm_path + f'train_scores_{i}.json', 'r') as f:
                sub_data = json.load(f)
                bert_top_n_data.update(sub_data)
            print(f'加载{nezha_wwm_path}' + f'train_scores_{i}.json')
            with open(nezha_wwm_path + f'train_scores_{i}.json', 'r') as f:
                sub_data = json.load(f)
                nezha_top_n_data.update(sub_data)
            print(f'加载{macbert_large_path}' + f'train_scores_{i}.json')
            with open(macbert_large_path + f'train_scores_{i}.json', 'r') as f:
                sub_data = json.load(f)
                macbert_top_n_data.update(sub_data)

            for q_id in tqdm(bert_top_n_data.keys(), desc='加载train_data_top_n_with_scores'):
                bert_top_n = bert_top_n_data[q_id]['top_n']
                nezha_top_n = nezha_top_n_data[q_id]['top_n']
                macbert_top_n = macbert_top_n_data[q_id]['top_n']
                temp_doc_list = []
                for bert, nezha, macbert in zip(bert_top_n, nezha_top_n, macbert_top_n):
                    score = (bert[2] + nezha[2] + macbert[2]) / 3
                    temp_doc_list.append((bert[0], bert[1], score))
                sorted_doc_list = sorted(temp_doc_list, key=lambda x: x[2], reverse=True)
                train_data_top_n_with_scores[q_id] = sorted_doc_list

        return train_data_top_n_with_scores

    '''
    选择{retriever}模型检索回的得分低于 0.1 的最高检索段落作为hard_negatives(只取前4个)作为新的检索训练集
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
    output_data_path = f'./dureader_data/retrieval_train_data_from_{retriever}/'
    train_path = './dureader_data/dureader_retrieval-data/train.json'

    train_file = open(train_path, 'r', encoding='utf-8')

    dual_train = []
    train_file_list = [json.loads(i) for i in train_file.readlines()]
    train_data_top_n_with_scores = merge_reranker_res()

    for item1 in tqdm(train_file_list, desc='构造dual_train.json'):
        qid = item1["question_id"]
        top_n_doc_list = train_data_top_n_with_scores[qid]
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
        # 选择得分低于 0.1 的最高检索段落（取前4条）
        for res in top_n_doc_list:
            if len(hard_negative_ctxs) >= 4:
                break
            if res[2] < 0.1 and res[0] not in answer_paragraphs_ids:
                hard_negative_ctxs.append({
                    "title": "",
                    "text": str(res[1])
                })
        if len(hard_negative_ctxs) < 4:
            print(f"问题 {qid} 的 hard_negatives 少于 4 条！")

        temp["hard_negative_ctxs"] = hard_negative_ctxs
        temp["positive_ctxs"] = positive_ctxs
        dual_train.append(temp)

    print(f"dual_train len：{len(dual_train)}")
    os.makedirs(output_data_path, exist_ok=True)
    with open(output_data_path + 'dual_train.json', 'w', encoding='utf-8') as f:
        json.dump(dual_train, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    get_hard_negatives_from_672_and_merge_pointwise_model()
