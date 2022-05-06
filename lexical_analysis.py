"""
Name : lexical_analysis.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/5/3 22:13
Desc: 用torch1.7环境跑
"""

'''
nz      其他专名
nw	    作品名
PER	    人名	
LOC	    地名	
ORG	    机构名	
TIME    时间
匹配问题中的以上实体，并要求文档一定包含问题中匹配到的实体，记录不符合条件的文档的以下信息（文档ID，出现的位置，失配比例（失配实体/总实体），是否正例）
'''

import json
from collections import defaultdict
from LAC import LAC
from tqdm import tqdm
import pandas as pd
import jieba
import os


# from ltp import LTP
# ltp = LTP()


def save_lac_jieba_result(data_with_pos_dict):
    if os.path.exists('./data/dureader_data/train_query_lac_jieba_res.json'):
        print(f'./data/dureader_data/train_query_lac_jieba_res.json 文件已存在，若想重新分词，请删除此文件。')
        return

    lac = LAC(mode='lac')
    label_sets = ['nz', 'nw', 'PER', 'LOC', 'ORG', 'TIME']
    # label_sets = ['PER', 'LOC', 'ORG', 'TIME']

    # 先用lac分词，再用jieba对label_sets中的实体分词
    q_lst = []  # 存储每条query的处理结果
    for qid in tqdm(data_with_pos_dict.keys()):
        query = data_with_pos_dict[qid]['query']
        lac_result = lac.run(query)
        word_list, lexical_list = lac_result[0], lac_result[1]
        label_word_sets = set()
        for word, label in zip(word_list, lexical_list):
            if label in label_sets and word not in label_word_sets:
                label_word_sets.add(word)
        wordpiece_dict = {}  # key是word，value是wordpiece list
        for word in label_word_sets:
            wordpiece = jieba.lcut(word, cut_all=True)
            wordpiece_dict[word] = wordpiece
        q_rel_doc_ids = [item['paragraph_id'] for item in data_with_pos_dict[qid]['answer_paragraphs']]
        q_lst.append({
            'qid': qid,
            'query': query,
            'lac_result': lac_result,
            'label_words': list(label_word_sets),
            'wordpiece': wordpiece_dict,
            'q_rel_doc_ids': q_rel_doc_ids,
        })
    with open('./data/dureader_data/train_query_lac_jieba_res.json', 'w', encoding='utf-8') as fw:
        json.dump(q_lst, fw, ensure_ascii=False, indent=4)


def save_lexical_analysis_result(recall_data_dict):
    with open('./data/dureader_data/train_query_lac_jieba_res.json', 'r', encoding='utf-8') as fr:
        train_query_lac_jieba_res = json.load(fr)

    # 需要记录的信息
    qid_lst, lac_result_lst, did_lst, rank_lst, q_count_lst, d_count_lst, mismatch_lst, is_pos_lst = [], [], [], [], [], [], [], []

    for item in tqdm(train_query_lac_jieba_res):
        qid = item['qid']
        lac_result = item['lac_result']
        label_words = item['label_words']
        wordpiece_dict = item['wordpiece']
        q_rel_doc_ids = item['q_rel_doc_ids']
        q_count = len(label_words)
        if q_count == 0:
            continue
        q_recall_docs = recall_data_dict[qid]
        for rank, (doc_id, doc_text) in enumerate(q_recall_docs):
            d_count = 0
            for word in label_words:
                for w in wordpiece_dict[word]:
                    if w.lower() in doc_text.lower():
                        d_count += 1
                        break
            mismatch = (q_count - d_count) / q_count
            is_pos = 1 if doc_id in q_rel_doc_ids else 0
            qid_lst.append(qid)
            lac_result_lst.append(lac_result)
            did_lst.append(doc_id)
            rank_lst.append(rank)
            q_count_lst.append(q_count)
            d_count_lst.append(d_count)
            mismatch_lst.append(mismatch)
            is_pos_lst.append(is_pos)

    pd.DataFrame({
        'qid': qid_lst,
        'lac_result': lac_result_lst,
        'did': did_lst,
        'rank': rank_lst,
        'q_count': q_count_lst,
        'd_count': d_count_lst,
        'mismatch': mismatch_lst,
        'is_pos': is_pos_lst
    }).to_csv('./lexical_analysis.tsv', index=False, sep='\t')


def get_pos_analysis():
    data = pd.read_csv('./lexical_analysis.tsv', delimiter='\t')
    i, j, k, l = 0, 0, 0, 0
    c, cs = 0, []
    for mismatch, is_pos in zip(data['mismatch'], data['is_pos']):
        if is_pos == 1:
            i += 1
            if mismatch == 0:
                j += 1
            elif mismatch < 1:
                k += 1
            elif mismatch <= 1:
                l += 1
                cs.append(c)
        c += 1

    data.iloc[cs].to_csv('./pos_lexical_analysis.tsv', index=False, sep='\t')

    print(i, j, k, l)


def get_train_data_info():
    train_data_with_pos_file = open('./data/dureader_data/dureader_retrieval-data/train.json', 'r', encoding='utf-8')
    train_data_with_pos = [json.loads(i) for i in train_data_with_pos_file.readlines()]
    # 处理成字典形式
    # ['question_id': {'question': '...', 'answer_paragraphs': [{'paragraph_id': '...', 'paragraph_text': '...'}]}]
    train_data_with_pos_dict = defaultdict(dict)
    for item in train_data_with_pos:
        qid = item['question_id']
        query = item['question']
        answer_paragraphs = item['answer_paragraphs']
        train_data_with_pos_dict[qid] = {'query': query,
                                         'answer_paragraphs': answer_paragraphs}

    recall_train_data_path = './0.566/train_data_top50.json'
    recall_train_data = json.load(open(recall_train_data_path, 'r', encoding='utf-8'))
    # 处理成字典形式
    # {'question_id': [(doc_id, doc_text), (...)]}
    recall_train_data_dict = defaultdict(dict)
    for item in recall_train_data:
        qid = item['q_id']
        top_n_docs = item['top_50']
        recall_train_data_dict[qid] = top_n_docs

    save_lac_jieba_result(train_data_with_pos_dict)
    save_lexical_analysis_result(recall_train_data_dict)
    get_pos_analysis()


def rerank_test_res(test_res_path):
    with open(test_res_path, 'r', encoding='utf-8') as f:
        test_res = json.load(f)



if __name__ == '__main__':
    get_train_data_info()
