
# coding: utf-8

# Siraj 的 README.md Credits 有交代，來源是 [pannouse repo](https://github.com/pannous/tensorflow-speech-recognition) 
# 裡面有一堆東西，將來可以一一玩玩學習學習，其中的 `lstm-tflearn.py` 正是他給 newcomers 的入門處。
#     
# 比起 Siraj 的 demo.py 這個要好一點點，有下列問題要解決：
# 
# - [ ] 1. while loop 也是亂寫
# - [ ] 2. Train data 跟 Test data 都是同一鍋 data 難怪要發生 overfitting
# - [ ] 3. training_iters = 300000 早就已經 overfitting 了 --> 如何用 tensorboard 觀察訓練狀況
# 
# 
# 放棄 Siraj 的 demo.py 改玩這個 lstm-tflearn.py

# In[1]:


#!/usr/bin/env python
#!/usr/bin/env python
import tensorflow as tf


# In[2]:


import tflearn


# In[4]:


import speech_data


# In[5]:


import time
import peforth

epoch_count = 10
learning_rate = 0.0001
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits


# In[6]:


batch = word_batch = speech_data.mfcc_batch_generator(batch_size)


# In[7]:



# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128*4, dropout=0.5)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)

## add this "fix" for tensorflow version errors
for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): tf.add_to_collection(tf.GraphKeys.VARIABLES, x )



# In[8]:


# Training
for i in range(1,int(training_iters/epoch_count)):
    trainX, trainY, testX, testY = next(batch)
    model.fit(
        trainX,
        trainY,
        n_epoch=epoch_count,
        validation_set=(testX, testY),
        show_metric=True,
        batch_size=batch_size)
    # must have the saved_networks/ folder
    training_step = i * epoch_count
    if training_step == 100: # I want to see it w/o a long wait
        time_stamp = int(time.time())
        model.save("saved_networks/tflearn.lstm.model."+str(time_stamp))
    if training_step % 2000 == 0:
        time_stamp = int(time.time())
        model.save("saved_networks/tflearn.lstm.model."+str(time_stamp))
        


# In[ ]:


peforth.ok()


# In[ ]:


_y = model.predict(next(batch)[0])  # << add your own voice here
print (_y) 

