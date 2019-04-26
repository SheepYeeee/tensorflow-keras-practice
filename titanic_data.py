import urllib.request
import os 

url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath="d:/py/tens_keras/data/titanic.xls"
#如果該路徑之檔案不存在就要下載文件
if not os.path.isfile(filepath): 
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)

import numpy 
import pandas as pd 

all_df=pd.read_excel(filepath)
#下面為我要的資料欄位 對應中文分別為:
#是否生還(label) 
#features=>名字 艙等 性別 年齡 手足或配偶也在船上的數量 
#雙親或子女也在船上的數量 旅客費用 登船港口(C,Q,S)
cols=['survived','name','pclass','sex','age','sibsp',
      'parch','fare','embarked']
all_df=all_df[cols]

#數據預處理
df = all_df.drop(['name'],axis=1)#刪除name字段

#找出含有null的字段，深度學習訓練時字段數據必須為數字，不得為null
all_df.isnull().sum()
age_mean=df['age'].mean()#年齡平均值
df['age'] = df['age'].fillna(age_mean)#將null替換成年齡平均值

fare_mean=df['fare'].mean()#旅客費用平均值
df['fare'] = df['fare'].fillna(fare_mean)#將null替換成旅客費用平均值
#將性別字段的文字轉為0或1
df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)

#將embarked(登船港口)轉換為三個字段(一個港口一個，若是該港口登船，值為1)
x_OneHot_df=pd.get_dummies(data=df,columns=['embarked'])

#dataframe轉Array
ndarray=x_OneHot_df.values
#提取features & label
Label = ndarray[:,0]#第一個冒號為提取所有項數 0為第0個字段 也就是label
Features = ndarray[:,1:]#第一個冒號為提取所有項數 1:為提取第一至最後字段

#將ndarray特徵字段進行標準化
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
scaledFeatures = minmax_scale.fit_transform(Features)
#print(scaledFeatures[:2])#標準化後的數字都介於0與1之間

#隨機將數據分為training data跟testing data
msk = numpy.random.rand(len(all_df))<0.8
train_df=all_df[msk]
test_df=all_df[~msk]
print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))
      
from lib import PreprocessData,show_train_history
# def PreprocessData(raw_df):
#     df=raw_df.drop({'name'},axis=1)
#     age_mean=df['age'].mean()
#     df['age']=df['age'].fillna(age_mean)
#     fare_mean=df['fare'].mean()
#     df['fare']=df['fare'].fillna(fare_mean)
#     df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)
#     x_OneHot_df=pd.get_dummies(data=df,columns=['embarked'])

#     ndarray=x_OneHot_df
#     Features = ndarray[:,1:]
#     Label = ndarray[:,0]

#     minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
#     scaledFeatures=minmax_scale.fit_transform(Features)

#     return scaledFeatures,Label

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)

print(train_Features[:2])
print(train_Label[:2])