from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential

import numpy as np
import pandas as pd

from lib import plot_images_labels_prediction, show_train_history

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

# 訓練資料預處理
x_train_one_dimension = x_train_image.reshape(60000, 784).astype('float32')
x_test_one_dimension = x_test_image.reshape(10000, 784).astype('float32')

x_train_normalize = x_train_one_dimension / 255
x_test_normalize = x_test_one_dimension / 255

# 標籤預處理

y_train_one_hot = np_utils.to_categorical(y_train_label)
y_test_one_hot = np_utils.to_categorical(y_test_label)

model = Sequential()

model.add(
    Dense(
        units=1000,
        input_dim=784,
        kernel_initializer='normal',
        activation='relu'
    )
)

model.add(
    Dense(
        units=10,
        kernel_initializer='normal',
        activation='softmax'
    )
)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_history = model.fit(
    x=x_train_normalize,
    y=y_train_one_hot,
    validation_split=0.2,
    epochs=10,
    batch_size=200,
    verbose=2
)

scores = model.evaluate(x_test_normalize, y_test_one_hot)
print(scores[1])

prediction = model.predict_classes(x_test_one_dimension)

plot_images_labels_prediction(
    x_test_image,
    y_test_label,
    prediction,
    250,
)

pd.crosstab(
    y_test_label,
    prediction,
    rownames=['label'],
    colnames=['predict']
)