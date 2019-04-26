from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
np.random.seed(10)

from lib import show_train_history

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

# 圖片預處理

x_train_4D = x_train_image.reshape(x_train_image.shape[0], 28, 28, 1).astype('float32')
x_test_4D = x_test_image.reshape(x_test_image.shape[0], 28, 28, 1).astype('float32')

x_train_4D_normalize = x_train_4D / 255
x_test_4D_normalize = x_test_4D / 255

# 標籤預處理

y_train_one_hot = np_utils.to_categorical(y_train_label)
y_test_one_hot = np_utils.to_categorical(y_test_label)

# 建模型

model = Sequential()

model.add(
    Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='same',
        input_shape=(28, 28, 1),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    Conv2D(
        filters=36,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(Dropout(0.25))

model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        10,
        activation='softmax'
    )
)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_history = model.fit(
    x=x_train_4D_normalize,
    y=y_train_one_hot,
    validation_split=0.2,
    epochs=10,
    batch_size=300,
    verbose=2
)

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

model.evaluate(x_test_4D_normalize, y_test_one_hot)
prediction = model.predict_classes(x_test_4D_normalize)

pd.crosstab(
    y_test_label,
    prediction,
    rownames=['label'],
    colnames=['predict']
)