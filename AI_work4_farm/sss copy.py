import os
import jieba
import re
import os
import sys
import time
from jieba import analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# 引入TF-IDF关键词抽取接口


path = "C:\\Users\\admin\\Desktop\\farming\\Train\\dataTrainComplete"  # 資料夾目錄
files = os.listdir(path)  # 得到資料夾下的所有檔名稱
s = []
for file in files:  # 遍歷資料夾
    if not os.path.isdir(file):  # 判斷是否是資料夾，不是資料夾才開啟
        # print(file)
        filename = "C:\\Users\\admin\\Desktop\\farming\\Train\\dataTrainComplete\\"+file
        # print(filename)
        f = open(filename, 'r', encoding="utf-8")  # 開啟檔案
        jieba.set_dictionary('dict.txt-v2.big')
        jieba.load_userdict(
            'C:\\Users\\admin\\Desktop\\farming\\key.txt')  # 自己字典
        jieba_results = []
        totalContent = []
        for sent in f.readlines():
            # 載入停用字
            stopwordset = set()
            with open('stopwords.txt', 'r', encoding='utf-8') as sw:
                for line in sw:
                    stopwordset.add(line.strip('\n'))
            words = jieba.cut(sent, cut_all=False)
            article = ''
            for word in words:
                # 正規表達式，只針對文字處理
                m = re.match(r'^[\u4E00-\u9FFFa-zA-Z]+$', word)
                if m is not None:
                    if word not in stopwordset:
                        article += word
                        article += ' '
            totalContent.append(article)
            print(totalContent)

        f.close()
