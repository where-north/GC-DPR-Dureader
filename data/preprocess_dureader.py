"""
Name : preprocess_dureader.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/21 11:27
Desc:
"""

import json
from tqdm import tqdm
import pandas as pd
import re


def get_dual_train():
    '''
    将./dureader_data/dureader-retrieval-baseline-dataset/train/dual.train.tsv
    格式：query null para_text_pos null para_text_neg null # 每个query有4个para_text_neg，若干个para_text_pos
    处理成 json：
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

    input_data_path = './dureader_data/dureader-retrieval-baseline-dataset/train/dual.train.tsv'
    output_data_path = './dureader_data/retrieval_train_data/dual_train.json'

    input_data = []
    with open(input_data_path, 'r', encoding='UTF-8') as f:
        for i, j in enumerate(f.readlines()):
            temp = j.strip().split('\t')
            input_data.append([temp[0].strip(), temp[2].strip(), temp[4].strip()])

    input_data_len = len(input_data)
    print(f'input len: {input_data_len}')

    output_data = []
    # 第一条数据
    single_train_data = {'question': input_data[0][0],
                         "answers": [],
                         "positive_ctxs": [{
                             "title": "",
                             "text": input_data[0][1]
                         }],
                         "negative_ctxs": [],
                         "hard_negative_ctxs": set()}
    for i in tqdm(range(input_data_len)):
        if i > 0 and i % 4 == 0:
            if input_data[i][0] != input_data[i - 1][0]:
                temp_cache = single_train_data['hard_negative_ctxs']
                single_train_data['hard_negative_ctxs'] = []
                for text in temp_cache:
                    single_train_data['hard_negative_ctxs'].append({'title': '', 'text': text})
                output_data.append(single_train_data)
                single_train_data = {'question': input_data[i][0],
                                     "answers": [],
                                     "positive_ctxs": [],
                                     "negative_ctxs": [],
                                     "hard_negative_ctxs": set()}
        if i % 4 == 0:
            single_train_data['positive_ctxs'].append({
                "title": "",
                "text": input_data[i][1]
            })
        single_train_data['hard_negative_ctxs'].add(input_data[i][2])

    print(f'output len: {len(output_data)}')
    print(output_data[0])
    with open(output_data_path, 'w', encoding='UTF-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def get_dual_dev():
    '''
    将./dureader_data/dureader-retrieval-baseline-dataset/dev/dev.json
    格式：[{'question_id': '...', 'question': '...', 'answer_paragraphs': [{'paragraph_id': '...', 'paragraph_text': '...'}, ...]}, ...]
    处理成 json：
    [
      {
       "dataset": "dureader_passage",
       "question": "....",
       "answers": ["...", "...", "..."],
       "negative_ctxs": [],
       "hard_negative_ctxs": [],
       "positive_ctxs": [{
          "title": "...",
          "text": "....",
          "score": "...",
          "title_score": "...",
          "passage_id": "..."
       }]
    ]
    '''

    input_data_path = './dureader_data/dureader-retrieval-baseline-dataset/dev/dev.json'
    output_data_path = './dureader_data/retrieval_train_data/dual_dev.json'

    output_data = []
    with open(input_data_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            temp = {"dataset": "dureader_passage", "answers": [], "negative_ctxs": [], "hard_negative_ctxs": []}
            data = json.loads(line)
            temp['question'] = data['question']
            temp['positive_ctxs'] = []
            for i in data['answer_paragraphs']:
                temp['positive_ctxs'].append({
                    "title": "",
                    "text": i['paragraph_text'],
                    "score": "",
                    "title_score": "",
                    "passage_id": i['paragraph_id']
                })
            output_data.append(temp)

    print(output_data[0])

    with open(output_data_path, 'w', encoding='UTF-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def get_passage():
    '''
    将 ./dureader_data/dureader_retrieval-data/passage_collection.json
    格式：
    {
       "data":[
          {
             "paragraph_id":"de1e2b18d9724d6a9373ac433e027886",
             "paragraph_text": "..."
          },
          "…",
          {
             "paragraph_id":"df897061476993dcc99558d0deac8387",
             "paragraph_text": "..."
          }
       ]
    }
    处理成 tsv：
    doc_id, doc_text, title
    '''

    RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                     u'|' + \
                     u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                     (chr(0xd800), chr(0xdbff), chr(0xdc00), chr(0xdfff),
                      chr(0xd800), chr(0xdbff), chr(0xdc00), chr(0xdfff),
                      chr(0xd800), chr(0xdbff), chr(0xdc00), chr(0xdfff),
                      )

    passage_collection_path = './dureader_data/dureader_retrieval-data/passage_collection.json'
    doc_id, doc_text, title = [], [], []
    with open(passage_collection_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            passage_id_text = json.loads(line)

            text = re.sub(r"[\x01-\x1F\x7F]", "", passage_id_text['paragraph_text'])
            text = re.sub(RE_XML_ILLEGAL, "", text).strip()
            if isinstance(text, str):
                doc_id.append(passage_id_text['paragraph_id'])
                doc_text.append(text)
                title.append('')

    assert len(doc_id) == len(doc_text) == len(title)

    pd.DataFrame({
        'id': doc_id,
        'text': doc_text,
        'title': title
    }).to_csv('./dureader_data/passages.tsv', index=False, sep='\t')

    # 下面这部分代码将passages的前一半分为4份
    # 4048334
    # chunksize返回一个迭代器，每次迭代读取返回chunksize大小的数据
    data = pd.read_csv('./dureader_data/passages.tsv', delimiter='\t', chunksize=4048334)
    for i, j in enumerate(data):
        if i == 0:
            sub_len = int(4048334 / 4) + 1
            for idx in range(4):
                start_idx = idx * sub_len
                end_idx = start_idx + sub_len
                print(f"split data from {start_idx} to {end_idx}: ")
                j.iloc[start_idx:end_idx].to_csv(f'./dureader_data/passages0-{idx}.tsv', index=False, sep='\t')
            break

    # 下面这部分代码将passages的后一半分为4份
    data = pd.read_csv('./dureader_data/passages.tsv', delimiter='\t', chunksize=4048334)
    for i, j in enumerate(data):
        if i == 1:
            sub_len = int(4048334 / 4) + 1
            for idx in range(4):
                start_idx = idx * sub_len
                end_idx = start_idx + sub_len
                print(f"split data from {start_idx} to {end_idx}: ")
                j.iloc[start_idx:end_idx].to_csv(f'./dureader_data/passages1-{idx}.tsv', index=False, sep='\t')
            break


def get_dual_dev2():
    '''
    为dev.json中的question检索回top50 paragraph，去除相关段落后挑选30条hard negatives，用于检索模型和重排序模型评估。
    dev.json格式：
    [{'question_id': '...', 'question': '...', 'answer_paragraphs': [{'paragraph_id': '...', 'paragraph_text': '...'}, ...]}, ...]
    检索回的top50 paragraph 文件格式：
    dev_data_top50.json:
       [
       {'q_text': '',
       'q_id': '',
       'top_50': [(doc_id, doc_text), (...)]}
       ]
    融合前两者成新的dev文件dev_with_hn.json：
    [
      {
       "dataset": "dureader_passage",
       "question": "....",
       "answers": ["...", "...", "..."],
       "negative_ctxs": [],
       "hard_negative_ctxs": [],
       "positive_ctxs": [{
          "title": "...",
          "text": "....",
          "score": "...",
          "title_score": "...",
          "passage_id": "..."
       }]
    ]
    '''
    dev_path = './dureader_data/dureader-retrieval-baseline-dataset/dev/dev.json'
    pre_dev_path = '../dev_data_top50.json'
    save_path = './dureader_data/retrieval_train_data/'
    dev_file = open(dev_path, 'r', encoding='utf-8')
    pre_dev_file = open(pre_dev_path, 'r', encoding='utf-8')
    dev_with_hn = []
    dev_file_list = [json.loads(i) for i in dev_file.readlines()]
    pre_dev_file_list = json.load(pre_dev_file)
    for item1, item2 in zip(dev_file_list, pre_dev_file_list):
        temp = {
            "dataset": "dureader_passage",
            "question": item1["question"],
            "answers": [],
            "negative_ctxs": [],
            "hard_negative_ctxs": [],
            "positive_ctxs": []
        }
        answer_paragraphs = item1["answer_paragraphs"]
        answer_paragraphs_ids = [i["paragraph_id"] for i in answer_paragraphs]
        positive_ctxs = [{"title": "",
                          "text": i["paragraph_text"],
                          "score": "",
                          "title_score": "",
                          "passage_id": i["paragraph_id"]} for i in answer_paragraphs]
        hard_negative_ctxs = []
        for res in item2["top_50"]:
            if len(hard_negative_ctxs) >= 30:
                break
            if res[0] not in answer_paragraphs_ids:
                hard_negative_ctxs.append({
                    "title": "",
                    "text": res[1],
                    "score": "",
                    "title_score": "",
                    "passage_id": res[0]
                })

        temp["hard_negative_ctxs"] = hard_negative_ctxs
        temp["positive_ctxs"] = positive_ctxs
        dev_with_hn.append(temp)

    print(f"dev_with_hn len：{len(dev_with_hn)}")
    with open(save_path + "dev_with_hn.json", 'w', encoding='utf-8') as f:
        json.dump(dev_with_hn, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # 转换训练集格式
    get_dual_train()
    # 转换验证集格式
    get_dual_dev()
    # 转换语料库格式
    get_passage()
