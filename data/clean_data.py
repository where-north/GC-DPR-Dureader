"""
Name : clean_data.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/30 19:18
Desc:
"""
import json
import pandas as pd
from tqdm import tqdm

dual_train_data_path = './dureader_data/retrieval_train_data_from_baseline/dual_train.json'
dual_dev_data_path = './dureader_data/retrieval_train_data_from_baseline/dev_with_hn.json'
passage_data_path = './dureader_data/passages{}-{}.tsv'

cleaned_dual_train_data_path = './dureader_data/retrieval_train_data_from_baseline/cleaned_dual_train.json'
cleaned_dual_dev_data_path = './dureader_data/retrieval_train_data_from_baseline/cleaned_dev_with_hn.json'
cleaned_passage_data_path = './dureader_data/cleaned_passages{}-{}.tsv'

dual_train_data = json.load(open(dual_train_data_path, 'r', encoding='utf-8'))
print(len(dual_train_data))
dual_dev_data = json.load(open(dual_dev_data_path, 'r', encoding='utf-8'))
print(len(dual_dev_data))
for item in tqdm(dual_train_data):
    for ctx in item['positive_ctxs']:
        ctx['text'] = ctx['text'].replace('百度经验:jingyan.baidu.com', '')
    for ctx in item['hard_negative_ctxs']:
        ctx['text'] = ctx['text'].replace('百度经验:jingyan.baidu.com', '')

for item in tqdm(dual_dev_data):
    for ctx in item['positive_ctxs']:
        ctx['text'] = ctx['text'].replace('百度经验:jingyan.baidu.com', '')
    for ctx in item['hard_negative_ctxs']:
        ctx['text'] = ctx['text'].replace('百度经验:jingyan.baidu.com', '')

with open(cleaned_dual_train_data_path, 'w', encoding='utf-8') as f1:
    json.dump(dual_train_data, f1, ensure_ascii=False, indent=4)

with open(cleaned_dual_dev_data_path, 'w', encoding='utf-8') as f2:
    json.dump(dual_dev_data, f2, ensure_ascii=False, indent=4)

for i in range(2):
    for j in range(4):
        data_path = passage_data_path.format(i, j)
        print(f'正在处理 {data_path}')
        passage_data = pd.read_csv(data_path, delimiter='\t')
        text_list = passage_data['text']
        id_list = passage_data['id']
        title_list = passage_data['title']
        cleaned_text_list = []
        for idx, text in enumerate(text_list):
            if str(type(text)) == "<class 'float'>":
                text_id = id_list.pop(idx)
                title_list.pop(idx)
                print(f'出现 {text} 值，删除 id 为 {text_id} 的文章。')
                continue
            cleaned_text_list.append(text.replace('百度经验:jingyan.baidu.com', ''))

        pd.DataFrame({
            'id': id_list,
            'text': cleaned_text_list,
            'title': title_list
        }).to_csv(cleaned_passage_data_path.format(i, j), index=False, sep='\t')



'''
日志：
正在处理 ./dureader_data/passages0-0.tsv
出现 nan 值，删除 id 为 e0815daa869fde85e659963196defb7b 的文章。
正在处理 ./dureader_data/passages0-1.tsv
正在处理 ./dureader_data/passages0-2.tsv
出现 nan 值，删除 id 为 84ff14fa45be3ca4739e7c027717a541 的文章。
正在处理 ./dureader_data/passages0-3.tsv
正在处理 ./dureader_data/passages1-0.tsv
正在处理 ./dureader_data/passages1-1.tsv
正在处理 ./dureader_data/passages1-2.tsv
正在处理 ./dureader_data/passages1-3.tsv
'''