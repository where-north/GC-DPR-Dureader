"""
Name : merge.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/5/26 18:26
Desc:
"""
import json
from tqdm import tqdm
from collections import defaultdict
import ijson


def norm_score(score_list):
    min_, max_ = min(score_list), max(score_list)
    diff = max_ - min_
    norm_score_list = []
    for sco in score_list:
        if diff > 0:
            norm_score_list.append((sco - min_) / diff)
        else:
            norm_score_list.append(1)
    return norm_score_list


def get_reranker_res():
    model_path = '0.771_0.784'
    main_path = f'/home/chy/reranker-main/reranker_model/{model_path}/'
    print(f'加载{main_path}')
    top_n_data = defaultdict(dict)
    for i in range(4):
        with open(main_path + f'reranker_scores_{i}.json', 'r') as f:
            sub_data = json.load(f)
            top_n_data.update(sub_data)

    """加载top50数据
    单条格式：
    'q_id':{'q_text': '',
    'top_n': [(doc_id, doc_text, doc_score), (...)]}
    """
    sort_res = defaultdict(list)
    for q_id in top_n_data.keys():
        top_n = top_n_data[q_id]['top_n']
        sort_top_n = sorted(top_n, key=lambda x: x[2], reverse=True)
        pid_list = [i[0] for i in sort_top_n]
        score_list = [i[2] for i in sort_top_n]
        norm_score_list = norm_score(score_list)
        sort_res[q_id] = [[i, j] for i, j in zip(pid_list, norm_score_list)]
    return sort_res


def get_dpr_res(data_path='./test_data_top200.json'):
    print(f'加载{data_path}')
    dic = defaultdict(list)
    with open(f'{data_path}', 'r', encoding='utf-8') as f:
        dpr_top_res = json.load(f)
    for item in dpr_top_res:
        dic[item['q_id']] = item['top_n']
    return dic


def get_dpr_data(data_path='./train_data_top200_with_scores.json'):
    print(f'加载{data_path}')
    dpr_top_res = defaultdict(dict)
    with open(data_path, 'r', encoding='utf-8') as f:
        # 数据格式：{'q_text': questions[i], 'q_id': qid,
        #                 'top_n': [(doc_id, passages[doc_id], np.float_(doc_score)) for doc_id, doc_score in
        #                           zip(doc_ids, norm_score_list)]}
        objects = ijson.items(f, 'item')
        for item in tqdm(objects, desc=f'加载{data_path}'):
            dpr_top_res[item['q_id']] = item

    return dpr_top_res


def merge_dpr_res():
    merge_result = {}
    dpr_top_res1 = get_dpr_res('./0.672/test_data_top200_with_scores.json')
    dpr_top_res2 = get_dpr_res('./0.693/test_data_top200_with_scores.json')

    for qid in tqdm(dpr_top_res1.keys()):
        pid_score_dict = {}
        for pid_score in dpr_top_res1[qid]:
            pid, score = pid_score[0], pid_score[2]
            pid_score_dict[pid] = score

        score_list = [i[2] for i in dpr_top_res2[qid]]
        pid_list = [i[0] for i in dpr_top_res2[qid]]
        for pid, score in zip(pid_list, score_list):
            if pid_score_dict.get(pid, -1) == -1:
                pid_score_dict[pid] = score
            else:
                pid_score_dict[pid] = pid_score_dict[pid] + score

        merge_pid_scores = [i for i in pid_score_dict.items()]
        merge_pid_scores = sorted(merge_pid_scores, key=lambda x: x[1], reverse=True)
        merge_top50_pids = [i[0] for i in merge_pid_scores][:50]
        merge_result[qid] = merge_top50_pids

    with open('./merge_result.json', 'w', encoding='utf-8') as f:
        json.dump(merge_result, f, indent=4, ensure_ascii=False)


def merge_dpr_train_data():
    dpr_top_train_data1 = get_dpr_data('./0.672/train_data_top200_with_scores.json')
    dpr_top_train_data2 = get_dpr_data('./0.693/train_data_top200_with_scores.json')
    passage_collection = defaultdict(str)
    merge_train_data = []
    for qid in tqdm(dpr_top_train_data1.keys()):
        pid_score_dict = {}
        for item in dpr_top_train_data1[qid]['top_n']:
            pid, p_text, score = item[0], item[1], item[2]
            pid_score_dict[pid] = score
            if pid not in passage_collection:
                passage_collection[pid] = str(p_text)

        for item in dpr_top_train_data2[qid]['top_n']:
            pid, p_text, score = item[0], item[1], item[2]
            if pid_score_dict.get(pid, -1) == -1:
                pid_score_dict[pid] = score
            else:
                pid_score_dict[pid] = pid_score_dict[pid] + score
            if pid not in passage_collection:
                passage_collection[pid] = str(p_text)

        merge_pid_scores = [i for i in pid_score_dict.items()]
        merge_pid_scores = sorted(merge_pid_scores, key=lambda x: x[1], reverse=True)
        merge_top200_pids = [i[0] for i in merge_pid_scores][:200]
        tmp_top_n = []
        for pid in merge_top200_pids:
            tmp_top_n.append((pid, str(passage_collection[pid])))

        merge_train_data.append({'q_text': dpr_top_train_data1[qid]['q_text'], 'q_id': qid,
                                 'top_n': tmp_top_n})

    with open('./train_data_top200.json', 'w', encoding='utf-8') as f:
        json.dump(merge_train_data, f, indent=4, ensure_ascii=False)


def merge_dpr_dev_data():
    dpr_top_dev_data1 = get_dpr_data('./0.672/dev_data_top50_with_scores.json')
    dpr_top_dev_data2 = get_dpr_data('./0.693/dev_data_top50_with_scores.json')
    passage_collection = defaultdict(str)
    merge_dev_data = []
    for qid in tqdm(dpr_top_dev_data1.keys()):
        pid_score_dict = {}
        for item in dpr_top_dev_data1[qid]['top_n']:
            pid, p_text, score = item[0], item[1], item[2]
            pid_score_dict[pid] = score
            if pid not in passage_collection:
                passage_collection[pid] = str(p_text)

        for item in dpr_top_dev_data2[qid]['top_n']:
            pid, p_text, score = item[0], item[1], item[2]
            if pid_score_dict.get(pid, -1) == -1:
                pid_score_dict[pid] = score
            else:
                pid_score_dict[pid] = pid_score_dict[pid] + score
            if pid not in passage_collection:
                passage_collection[pid] = str(p_text)

        merge_pid_scores = [i for i in pid_score_dict.items()]
        merge_pid_scores = sorted(merge_pid_scores, key=lambda x: x[1], reverse=True)
        merge_top200_pids = [i[0] for i in merge_pid_scores][:50]
        tmp_top_n = []
        for pid in merge_top200_pids:
            tmp_top_n.append((pid, str(passage_collection[pid])))

        merge_dev_data.append({'q_text': dpr_top_dev_data1[qid]['q_text'], 'q_id': qid,
                               'top_n': tmp_top_n})

    with open('./dev_data_top50.json', 'w', encoding='utf-8') as f:
        json.dump(merge_dev_data, f, indent=4, ensure_ascii=False)


def merge_dpr_test_data():
    dpr_top_test_data1 = get_dpr_data('./0.672/test_data_top200_with_scores.json')
    dpr_top_test_data2 = get_dpr_data('./0.693/test_data_top50_with_scores.json')
    passage_collection = defaultdict(str)
    merge_test_data = []
    for qid in tqdm(dpr_top_test_data1.keys()):
        pid_score_dict = {}
        for item in dpr_top_test_data1[qid]['top_n'][:50]:
            pid, p_text, score = item[0], item[1], item[2]
            pid_score_dict[pid] = score
            if pid not in passage_collection:
                passage_collection[pid] = p_text

        for item in dpr_top_test_data2[qid]['top_n']:
            pid, p_text, score = item[0], item[1], item[2]
            if pid_score_dict.get(pid, -1) == -1:
                pid_score_dict[pid] = score
            else:
                pid_score_dict[pid] = pid_score_dict[pid] + score
            if pid not in passage_collection:
                passage_collection[pid] = str(p_text)

        merge_pid_scores = [i for i in pid_score_dict.items()]
        merge_pid_scores = sorted(merge_pid_scores, key=lambda x: x[1], reverse=True)
        merge_top200_pids = [i[0] for i in merge_pid_scores][:50]
        tmp_top_n = []
        for pid in merge_top200_pids:
            tmp_top_n.append((pid, passage_collection[pid]))

        merge_test_data.append({'q_text': dpr_top_test_data1[qid]['q_text'], 'q_id': qid,
                               'top_n': tmp_top_n})

    with open('./test_data_top50.json', 'w', encoding='utf-8') as f:
        json.dump(merge_test_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    merge_dpr_train_data()
    merge_dpr_dev_data()
    merge_dpr_test_data()
