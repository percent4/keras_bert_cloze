# -*- coding: utf-8 -*-
# @Time : 2021/1/8 11:34
# @Author : Jclian91
# @File : cloze_predict.py
# @Place : Yangpu, Shanghai
import numpy as np
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint

# 加载词典
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'
token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

id_token_dict = {v: k for k, v in token_dict.items()}


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')
        return R


tokenizer = OurTokenizer(token_dict)

# 加载模型
model_path = "./chinese_L-12_H-768_A-12/"
bert_model = load_trained_model_from_checkpoint(
    model_path + "bert_config.json",
    model_path + "bert_model.ckpt",
    training=True
)
# bert_model.summary()


# 完形填空，预测MASK的字符
def get_mask_character(start_string, mask_num, end_string):
    string = list(start_string) + ['MASK'] * mask_num + list(end_string)
    token_ids, segment_ids = tokenizer.encode(string, max_len=512)
    for i in range(mask_num):
        token_ids[len(start_string)+i+1] = tokenizer._token_dict['[MASK]']

    # mask
    masks = [0] * 512
    for i in range(mask_num):
        masks[len(start_string)+i+1] = 1

    # 模型预测被mask掉的部分
    predicts = bert_model.predict([np.array([token_ids]), np.array([segment_ids]), np.array([masks])])[0]
    pred_indice = predicts[0][len(start_string)+1:len(start_string)+mask_num+1].argmax(axis=1).tolist()
    return [id_token_dict[_] for _ in pred_indice]


if __name__ == '__main__':
    # 原句1： 白云山，位于广东省广州市白云区，为南粤名山之一，自古就有“羊城第一秀”之称。
    start_str1 = "白云山，位于"
    end_str1 = "广州市白云区，为南粤名山之一，自古就有“羊城第一秀”之称。"
    pred_chars = get_mask_character(start_str1, 3, end_str1)
    print(pred_chars)

    # 原句2：首先，从市值看，腾讯和阿里市值已经有2500亿，而百度才500多亿，是BAT体量中最小的一家公司。
    start_str2 = "首先，从"
    end_str2 = "看，腾讯和阿里市值已经有2500亿，而百度才500多亿，是BAT体量中最小的一家公司。"
    pred_chars = get_mask_character(start_str2, 2, end_str2)
    print(pred_chars)

    # 原句3：特斯拉CEO埃隆·马斯克的个人净资产升至1850亿美元，超越亚马逊CEO贝索斯荣登全球第一大富豪。
    start_str3 = "特斯拉CEO埃隆·马斯克的个人净资产升至1850亿美元，超越亚马逊CEO贝索斯荣登"
    end_str3 = "第一大富豪。"
    pred_chars = get_mask_character(start_str3, 2, end_str3)
    print(pred_chars)

    # 原句4：我在上海闵行区工作。
    start_str4 = "我在上海闵"
    end_str4 = "区工作。"
    pred_chars = get_mask_character(start_str4, 1, end_str4)
    print(pred_chars)

    # 原句5: 坐地铁几号线可以到北京首都国际机场？
    start_str5 = "坐地铁几号线可以到北京"
    end_str5 = "国际机场？"
    pred_chars = get_mask_character(start_str5, 2, end_str5)
    print(pred_chars)