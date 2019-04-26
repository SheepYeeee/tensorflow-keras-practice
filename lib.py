import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy 
import pandas as pd 
from sklearn import preprocessing
import urllib.request
import os 
import re 

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):

    fig= plt.gcf()
    fig.set_size_inches(12, 14)
    
    if num > 25:
        num = 25
    
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=%s" % str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=%s" % str(prediction[idx])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(
        ['train', 'validation'],
        loc='upper left'
    )
    plt.show()


def PreprocessData(raw_df):
    df=raw_df.drop({'name'},axis=1)
    age_mean=df['age'].mean()
    df['age']=df['age'].fillna(age_mean)
    fare_mean=df['fare'].mean()
    df['fare']=df['fare'].fillna(fare_mean)
    df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)
    x_OneHot_df=pd.get_dummies(data=df,columns=['embarked'])

    ndarray=x_OneHot_df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaledFeatures=minmax_scale.fit_transform(Features)

    return scaledFeatures,Label

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