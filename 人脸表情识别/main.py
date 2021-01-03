import os
from idlelib import history

import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_sample = []
train_result = []
test_sample = []
test_result = []

dictionary = os.path.join('./train', 'angry')
picture_names = os.listdir(dictionary)
for i in range(len(picture_names)):
    picture = cv2.imread(os.path.join(dictionary, picture_names[i]), 0)
    picture = picture[:, :, np.newaxis]
    train_sample.append(picture)
    result = keras.utils.to_categorical(0, num_classes=7)
    train_result.append(result)
dictionary = os.path.join('./train', 'disgust')
picture_names = os.listdir(dictionary)
for i in range(len(picture_names)):
    picture = cv2.imread(os.path.join(dictionary, picture_names[i]), 0)
    picture = picture[:, :, np.newaxis]
    train_sample.append(picture)
    result = keras.utils.to_categorical(1, num_classes=7)
    train_result.append(result)
dictionary = os.path.join('./train', 'fear')
picture_names = os.listdir(dictionary)
for i in range(len(picture_names)):
    picture = cv2.imread(os.path.join(dictionary, picture_names[i]), 0)
    picture = picture[:, :, np.newaxis]
    train_sample.append(picture)
    result = keras.utils.to_categorical(2, num_classes=7)
    train_result.append(result)
dictionary = os.path.join('./train', 'happy')
picture_names = os.listdir(dictionary)
for i in range(len(picture_names)):
    picture = cv2.imread(os.path.join(dictionary, picture_names[i]), 0)
    picture = picture[:, :, np.newaxis]
    train_sample.append(picture)
    result = keras.utils.to_categorical(3, num_classes=7)
    train_result.append(result)
dictionary = os.path.join('./train', 'neutral')
picture_names = os.listdir(dictionary)
for i in range(len(picture_names)):
    picture = cv2.imread(os.path.join(dictionary, picture_names[i]), 0)
    picture = picture[:, :, np.newaxis]
    train_sample.append(picture)
    result = keras.utils.to_categorical(4, num_classes=7)
    train_result.append(result)
dictionary = os.path.join('./train', 'sad')
picture_names = os.listdir(dictionary)
for i in range(len(picture_names)):
    picture = cv2.imread(os.path.join(dictionary, picture_names[i]), 0)
    picture = picture[:, :, np.newaxis]
    train_sample.append(picture)
    result = keras.utils.to_categorical(5, num_classes=7)
    train_result.append(result)
dictionary = os.path.join('./train', 'surprise')
picture_names = os.listdir(dictionary)
for i in range(len(picture_names)):
    picture = cv2.imread(os.path.join(dictionary, picture_names[i]), 0)
    picture = picture[:, :, np.newaxis]
    train_sample.append(picture)
    result = keras.utils.to_categorical(6, num_classes=7)
    train_result.append(result)

train_sample = np.array(train_sample)
train_result = np.array(train_result)

df = pd.read_csv('./submission.csv')
for index, i in df.iterrows():
    file_name = i['file_name']
    picture = cv2.imread(os.path.join('./test', file_name), 0)
    picture = picture[:, :, np.newaxis]
    test_sample.append(picture)

test_sample = np.array(test_sample)

batch_size = 512
epochs = 100  # epochs = 20

model = Sequential()


def train():
    # 第一层卷积层
    model.add(Conv2D(input_shape=(48, 48, 1), filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # 第二层卷积层
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    # 第三层卷积层
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Flatten())

    # 全连接层
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))


def test():
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.fit(train_sample, train_result, batch_size=batch_size, epochs=epochs)
    predictions = model.predict_classes(test_sample)
    df['class'] = pd.DataFrame(predictions)
    df['class'] = df['class'].apply(lambda x: 'angry' if x == 0 else x)
    df['class'] = df['class'].apply(lambda x: 'disgust' if x == 1 else x)
    df['class'] = df['class'].apply(lambda x: 'fear' if x == 2 else x)
    df['class'] = df['class'].apply(lambda x: 'happy' if x == 3 else x)
    df['class'] = df['class'].apply(lambda x: 'neutral' if x == 4 else x)
    df['class'] = df['class'].apply(lambda x: 'sad' if x == 5 else x)
    df['class'] = df['class'].apply(lambda x: 'surprise' if x == 6 else x)
    df.to_csv('./result.csv', index=False)
    print(train_sample.shape)


def analyse():
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs2 = range(1, epochs + 1)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    ax.plot(epochs2, loss, 'bo', label='Training Loss')
    ax.plot(epochs2, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    ax2 = fig.add_subplot(122)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    ax2.plot(epochs2, acc, 'bo', label='Training Acc')
    ax2.plot(epochs2, val_acc, 'b', label='Validation Acc')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('./result.png')
    plt.show()


def main():
    train()
    test()
    analyse()


if __name__ == '__main__':
    main()
