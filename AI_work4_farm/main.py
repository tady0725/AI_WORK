import os
import jieba
import numpy as np
import pandas as pd
from jieba import analyse
from scipy.sparse import data
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import random


# 引入TF-IDF关键词抽取接口

allweight = []
path = "D:\\all_work\\farming\\Train\\dataTrainComplete"  # 資料夾目錄
files = os.listdir(path)  # 得到資料夾下的所有檔名稱
s = []
for file in files:  # 遍歷資料夾
    if not os.path.isdir(file):  # 判斷是否是資料夾，不是資料夾才開啟
        # print(file)
        filename = "D:\\all_work\\farming\\Train\dataTrainComplete\\"+file
        # print(filename)
        f = open(filename, 'r', encoding="utf-8")  # 開啟檔案

        jieba.load_userdict(
            'D:\\all_work\\farming\\key.txt')  # 自己字典
        s = []
        for sentence in f.readlines():

            seg_list = jieba.cut(sentence)
            seg_list = '/'.join(seg_list)
            # print(seg_list)
            s.append(seg_list.replace("\n", ""))  #

            for seg in s:
                if seg in '':
                    s.remove(seg)

        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        X = vectorizer.fit_transform(s)
        tfidf = transformer.fit_transform(X)
        weight = tfidf.toarray()
        # print(weight.shape)
        # print(tfidf[0])
        # print(weight)
        d = (list(chain.from_iterable(weight)))
        allweight.append(d)
        # print(d)
        # print("============================")
        f.close()
        # print('/'.join(seg_list))
        # print(seg_list)
# print([allweight[0]])
# x = random.randint(1, 561)
# y = random.randint(1, 561)
score = []
cc = 0
f1 = open("D:\\all_work\\farming\\submission_example.csv",
          'w', encoding="utf-8")
f1.write("Test,Reference")
f1.write("\n")
for i in range(len(allweight)):
    #b = list(map(float, allweight[i]))
    for j in range(i+1, len(allweight)):
        b = list(map(float, allweight[i]))
        c = list(map(float, allweight[j]))
        # Dot and norm
        dot = sum(a*b for a, b in zip(b, c))
        norm_a = sum(a*a for a in b) ** 0.5
        norm_b = sum(b*b for b in c) ** 0.5
        # Cosine similarity
        cos_sim = dot / (norm_a*norm_b)
        if cos_sim > 0.5:
            cc += 1

            f1.write(str(i)+","+str(j))
            f1.write("\n")
            # print(i, j)
print(cc)
f1.close()


# print('My version:', cos_sim)

'''
        keywords = analyse.extract_tags(seg_list, topK=10,
                                        withWeight=True, allowPOS=())
        a = []
        for keyword in keywords:
            a.append(keyword)
          # print(keyword)
        print(a)
        '''
