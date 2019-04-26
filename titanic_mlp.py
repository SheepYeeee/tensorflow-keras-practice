import urllib.request
import os 
import tarfile 
from lib import PreprocessData,show_train_history
import numpy
import pandas as pd

url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath="d:/py/tens_keras/data/titanic.xls"
#如果該路徑之檔案不存在就要下載文件
if not os.path.isfile(filepath): 
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
all_df=pd.read_excel(filepath)
# all_df=pd.read_excel("d:/py/tens_keras/data/titanic.xls",encoding='gb18030')

cols=['survived','name','pclass','sex','age','sibsp',
      'parch','fare','embarked']
all_df=all_df[cols]

msk=numpy.random.rand(len(all_df))<0.8
train_df=all_df[msk]
test_df=all_df[~msk]

print('total:',len(all_df),'train:',len(train_df),'test',len(test_df))

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential() #建立一個線性堆疊的模型
model.add(Dense(units=40,input_dim=9,
                kernel_initializer='uniform',
                activation='relu'))

model.add(Dense(units=30,
                kernel_initializer='uniform',
                activation='relu'))              

model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

#定義訓練方式
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])

#開始訓練
train_history=model.fit(x=train_Features,
                        y=train_Label,
                        validation_split=0.1,
                        epochs=30,
                        batch_size=30,verbose=2)

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

scores=model.evaluate(x=test_Features,
                      y=test_Label)
print(scores[1])

#加入jack跟rose的數據
Jack=pd.Series([0,'Jack',3,'male',23,1,0,5.0000,'5'])
Rose=pd.Series([1,'Rose',1,'female',20,1,0,100.0000,'5'])
JR_df=pd.DataFrame([list(Jack),list(Rose)],
                   columns=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked'])
all_df=pd.concat([all_df,JR_df])
print(all_df[-2:])


#進行預測  這邊開始怪怪的 但在jupyter可正常執行
all_Features,Label=PreprocessData(all_df)
all_probability=model.predict(all_Features)
print(all_probability[:10])

pd=all_df
pd.insert(len(all_df.columns),
          'probability',all_probability)
print(pd[-2:])

#生存率高 卻沒有存活的
print(pd[(pd['survived']==0) &  (pd['probability']>0.9) ])