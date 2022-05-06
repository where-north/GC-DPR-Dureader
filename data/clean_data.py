"""
Name : clean_data.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/4/30 19:18
Desc: remove empty passages
"""
import json
import pandas as pd
from tqdm import tqdm

passage_data_path = './dureader_data/passages{}-{}.tsv'
cleaned_passage_data_path = './dureader_data/cleaned_passages{}-{}.tsv'

all_text_list = []
all_id_list = []
all_title_list = []

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
            cleaned_text_list.append(text)

        pd.DataFrame({
            'id': id_list,
            'text': cleaned_text_list,
            'title': title_list
        }).to_csv(cleaned_passage_data_path.format(i, j), index=False, sep='\t')

        all_id_list.extend(id_list)
        all_text_list.extend(cleaned_text_list)
        all_title_list.extend(title_list)

pd.DataFrame({
    'id': all_id_list,
    'text': all_text_list,
    'title': all_title_list
}).to_csv('./dureader_data/cleaned_passages.tsv', index=False, sep='\t')

print(f'{len(all_id_list)}')

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
8096666
'''
