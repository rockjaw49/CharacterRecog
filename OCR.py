from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.datasets import mnist
import numpy as np

data = pd.read_csv(r"C:\OCRProject\A_Z Handwritten Data.csv").astype('float32')

print(data.head(10))

X = data.drop('0', axis = 1)
y = data['0']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6)
X_train = np.reshape(X_train.values, (X_train.shape[0], 28, 28))
X_test = np.reshape(X_test.values, (X_test.shape[0], 28, 28))

print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)

for i in range(9):  
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128)

scores = model.evaluate(X_test, y_test, verbose=1)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
model.summary()
model.save(r'model2.h5')