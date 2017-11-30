from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import tensorflow as tf
import peforth

learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

# Network building
net0 = net = tflearn.input_data([None, width, height])
net1 = net = tflearn.lstm(net, 128, dropout=0.8)
net2 = net = tflearn.fully_connected(net, classes, activation='softmax')
net3 = net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

### add this "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 


model = tflearn.DNN(net, tensorboard_verbose=0)

# https://github.com/llSourcell/tensorflow_speech_recognition_demo/issues/12
# You have to replace while 1: #training_iters: to for i in range(training_iters):
# while 1: #training_iters
for i in range(training_iters):
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)
