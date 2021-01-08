# -*- coding: utf-8 -*-
# @Time : 2021/1/8 12:24
# @Author : Jclian91
# @File : correct_test.py
# @Place : Yangpu, Shanghai
# 该脚本使用BERT的mask技术进行文本纠错
from cloze_predict import get_mask_character

sentence = "我要去埃及金子塔玩。"  # 金子塔中的子为错别字
sentence = "白云山，位于广东省广州市白云区，为南粤名山之一，自古就有“羊城第一秀”只称。"  # 只称中的只为错别字
sentence = "请把这个快递送到上海市闵航区。"  # 闵航区中的航为错别字
sentence = "少先队员因该为老人让坐"  # 因该中的因为错别字
sentence = "随然今天很热"  # 随然中的随为错别字
sentence = "我生病了,咳数了好几天"  # 咳数中的数为错别字
sentence = "《这就是铁甲》郑爽成铁甲女超能手，成功在经理人中脱引而出"  # 脱引而出中的引为错别字
sentence = "一群罗威纳犬宝宝打架，场面感忍。"  # 感忍中的忍为错别字
wrong_char_index = sentence.index("忍")

for i in range(len(sentence)):
    if i == wrong_char_index:
        start_string = sentence[:i]
        end_string = sentence[i+1:]
        pred_char = get_mask_character(start_string, 1, end_string)
        print("correct char: {}, predict char: {}".format(sentence[i], pred_char[0]))