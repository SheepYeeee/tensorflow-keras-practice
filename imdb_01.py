import urllib.request
import os 
import tarfile 
from lib import PreprocessData,show_train_history
import numpy
import pandas as pd
import tarfile

url="http://ai.stanford.edu/~amaas/data/sentiment"
filepath="d:/py/tens_keras/data/aclImdb_vl.tar.gz"
#如果該路徑之檔案不存在就要下載文件
if not os.path.isfile(filepath): 
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
#解壓縮
if not os.path.exists("d:/py/tens_keras/data/aclImdb"):
    tfile=tarfile.open("d:/py/tens_keras/data/aclImdb_vl.tar.gz",'r:*')
    result=tfile.extractall('d:/py/tens_keras/data/')#解壓縮後丟到這個路徑
#不建議用python載 檔案太大了 一直出錯 自己載自己解壓縮最保險

#資料準備
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re
def rm_tags(text):
    re_tag=re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

def read_files(filetype):
    path="d:/py/tens_keras/data/aclImdb/"
    file_list=[]

    positive_path=path+filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]

    negative_path=path+filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
    
    print('read',filetype,'files:',len(file_list))

    all_labels=([1]*12500+[0]*12500)
    all_texts=[]
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts+=[rm_tags(" ".join(file_input.readlines()))]
    
    return all_labels,all_texts

y_train,train_text=read_files("train")
y_test,test_text=read_files("test")

#讀取所有文章建立字典 限制字典的數量為nb_words=2000
token=Tokenizer(num_words=2000)#建立一個有2000個單字的字典
token.fit_on_texts(train_text)#讀取所有訓練資料 按照單字在影評中的出現次數進行排序 前2000個寫入字典中
print(token.document_count)
print(token.word_index)
#再將字典內的文字轉為數字列表
x_train_seq=token.texts_to_sequences(train_text)
x_test_seq=token.texts_to_sequences(test_text)
#讓轉換後的數字長度相同
x_train=sequence.pad_sequences(x_train_seq,maxlen=100)
x_test=sequence.pad_sequences(x_test_seq,maxlen=100)
