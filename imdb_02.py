from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb 
from lib import read_files

y_train,train_text=read_files("train")
y_test,test_text=read_files("test")
#建立token
token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)
#將影評文字轉成數字列表
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
#將數字列表的長度統一為100
x_train = sequence.pad_sequences(x_train_seq,maxlen=100)
x_test = sequence.pad_sequences(x_test_seq,maxlen=100)

#加入嵌入層 將數字列表轉乘向量列表
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding 

#建立線性堆疊模型
model = Sequential() 
#將嵌入層加入模型中
model.add(Embedding(output_dim=32,#將數字列表轉換為32維的向量
                    input_dim=2000,#2000個單字
                    input_length=100))#數字列表長度為100
model.add(Dropout(0.2))#加入dropout避免過度擬合(每次訓練會隨機在神經網路中20%的神經元 以避免過度擬合)

#建立多層感知模型
#將平坦層加入模型
model.add(Flatten())
#將隱藏層加入模型
model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(0.35))
#將輸出層加入模型
model.add(Dense(units=1,
                activation='sigmoid'))
print(model.summary())

#定義訓練方式
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])

#開始訓練#每一批100項數據#執行10個週期 顯示訓練過程
train_history = model.fit(x_train,y_train,batch_size=100,
                          epochs=10,verbose=2,
                          validation_split=0.2)

#評估模型的準確率
scores = model.evaluate(x_test,y_test,verbose=1)
print(scores[1])

#進行預測 (二維數組)
predict = model.predict_classes(x_test)
#將二維轉一維
predict_classes = predict.reshape(-1)
print(predict_classes[-20:])
#查看測試數據預測結果
SentmentDict={1:'正面的',0:'負面的'}
def display_test_SentmentDict(i):
    print(test_text[i])
    print('label真實值:',SentmentDict[y_test[i]],
           '預測結果:',SentmentDict[predict_classes[i]])

print(display_test_SentmentDict(12502))

#查看us的影評
input_text = '''
This film is what a lot of horror movies are missing today. The deep messages in this film. Each single shot of each single frame by Jordan Peele symbolizes everything. The film is very in depth and it's a conceptual horror film. It's what scares me the most. Jump scares and tension isn't enough and that's almost what every single predictable horror film does. This film captures the uncomfortable feeling of not knowing what to expect while adding layers upon layers in the process. The fact the movie doesn't have a predictable bland jump scare fest that moviesgo for us amazing. This film is honestly comparison to the shinning. So many details layers upon layers spreaded around. The real horror is us, as people. Can you trust that you can count on us? I left the theater thinking about it so much. So many horror elements that crept me out. This is the type of film that will be looked back at years and years to come by now. Jordan Peele trancends horror yet again. Hands down the greatest movie of 2019 so far. Lead actress deserves an Oscar. Not to mention the whole casts acting was insane. Screenplay was amazing. Sound crept under my skin and made me terrified of my favorite rap songs. I will not be listening to them the same way as before. The script is amazing. The plot twist. This movie has it all and more. It can be too much for people and it can be confusing. And that's fine. Because with Jordan Peele ever since Get Out... not everything is everyone's cup of tea (pun intended) but in all honesty, US Is a film that leaps boundaries and Isn't afraid to embrace the tricks up its sleeve and Jordan Peele proves it. I honestly believe this film is a masterpiece and Jordan Peele pulls it off yet again. I'm over 20 years old and I left the film terrified. The conseptual horror is always more terrifying than almost any other horror movie that Hollywood tries to give us. It's like audiences don't want to be challenged. They want a predictable story some jump scares and a happy ending and they'll give it a good review. This film was completely off the wall and challenges the viewer with so much. 10/10 and that's my honest opinion. This movie has so much rewatch value. Just like the shining had. I will be watching this film in years to come. Movies like this holds up compared to other horror movies because you can't run away from the conseptual horror. But other films you can. Once you timed the jump scare and know the plot and figure it out the rewatch value is 0/10. US on the other hand has so much rewatch value. If I watch it again I'd probably end up with much more questions than answers and that's even more terrifying. So much conversation and things going on. I'm out of words.
'''
#將影評文字轉為數字列表
input_seq = token.texts_to_sequences([input_text])
#將數字列表長度統一為100
pad_input_seq = sequence.pad_sequences(input_seq,maxlen=100)
#使用多層感知模型開始預測
predict_result=model.predict_classes(pad_input_seq)
print(SentmentDict[predict_result[0][0]])

#將預測影評的函數整理出來
def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq,maxlen=100)
    predict_result = model.predict_classes(pad_input_seq)
    print(SentmentDict[predict_result[0][0]])

