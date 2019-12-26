import pickle
import numpy as np

data_file = open('train_data.pickle','rb')
target_file = open('train_target.pickle','rb')

train_data = pickle.load(data_file)
train_target = pickle.load(target_file)

arr = [0,1,2,3,4,5]
t_target = np.array(arr)
print(t_target)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(28,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(len(train_target),activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

face_dic={'Diamond':0,'Oblong':1,'Oval':2,'Round':3,'Square':4,'Triangle':5}

from keras.utils import to_categorical

t_target = to_categorical(t_target)
print(t_target)

model.fit(train_data,t_target,epochs=100)

model.save('model.h5')



