from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import tensorflow as tf
import peforth
import time

epoch_count = 10
learning_rate = 0.0001
training_iters = 300000 # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)

# Network building
net0 = net = tflearn.input_data([None, width, height])
net1 = net = tflearn.lstm(net, 128, dropout=1.0) # 確定有 overfitting 才下 dropout
net2 = net = tflearn.fully_connected(net, classes, activation='softmax')
net3 = net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

### add this "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 

model = tflearn.DNN(net, tensorboard_verbose=0)


msg = '''
    dup 
    :> [0] constant loc // ( -- dict ) locals of the debugee
    :> [1] constant glo // ( -- dict ) globals of the debugee
    <text>

    Now load the network manually, e.g:
    model :: load("saved_networks/demo4-1512557901")
    
    </text> . '''
peforth.ok('00> ',glo=globals(),loc=locals(),cmd='cr')    
peforth.ok('11> ',glo=globals(),loc=locals(),cmd=msg)    

for i in range(1,int(training_iters/epoch_count)):
    trainX, trainY = next(batch)
    testX, testY = next(batch)
    model.fit(
        trainX,
        trainY,
        n_epoch=epoch_count,
        validation_set=(testX, testY),
        show_metric=True,
        batch_size=batch_size)
    training_step = i * epoch_count
    if training_step % 2000 == 0:
        # 事先要手動建立 saved_networks/ folder 
        time_stamp = int(time.time())
        model.save("saved_networks/tflearn.lstm.model."+str(time_stamp))
        # peforth.ok('22> ',loc=locals(),cmd='cr')    
        
peforth.ok('99> ',loc=locals(),cmd='cr')    
_y = model.predict(next(batch)[0])  # << add your own voice here
print (_y)

'''
    \ UUT 注意要領：
        [ ] 跑 demo2.py 以及 itchat bot ～\GitHub\peforth\playground\itchat\itchat_remote_peforth.py 
        [ ] power saving setting AC 'never' sleep 

    \ 完整設定過程，讓 UUT 回覆它的畫面經由 itchat bot 傳給遠端的 PC 或手機。
    \ 讓遠端可以來監看執行狀況。這段程式是由遠端灌過來給 UUT 的。
        \ 取得 itchat module object. UUT 的 peforth 是
        py> sys.modules['itchat'] constant itchat // ( -- module ) WeChat automation
        \ 取得 PIL 圖像處理工具
        import PIL.ImageGrab constant im // ( -- module ) PIL.ImageGrab
        \ 用部分 nickName 取得 chatroom object 
        itchat :> search_chatrooms('AILAB')[0] constant ailab // ( -- obj ) AILAB chatroom object
        \ 定義 check command :
        import time constant time // ( -- module )
        time :> ctime() . cr \ print now time
        : check ( -- ) // check UUT
            time :> ctime() . cr \ print now time
            im :: grab().save("1.jpg") \ Windows 顯示比例不要改，100% 就是全螢幕。
            ailab :> send("@img@1.jpg") \ 結果都送到 chatroom 
            . cr ;
        \ 完成設定，可以從遠端下達 @陳厚成0922 check 命令取得 UUT 畫面了。

    \ Breakpoint 11> 22> 等處都不要改，永久配合以下筆記
    11> depth . cr
    1
    11> :> [0] inport
    11> .s
    empty

    11> words
    ... snip ....
    (pyclude) pyclude .members .source dos cd --- __name__ __doc__ __package__ 
    __loader__ __spec__ __annotations__ __builtins__ __file__ __cached__ division 
    print_function absolute_import tflearn speech_data tf peforth learning_rate 
    training_iters batch_size width height classes batch word_batch net0 net net1 
    net2 net3 col x model

    11> batch type . cr    \ 是個 generator, next(batch) 產生 batch_size 組資料 tuple
    <class 'generator'>    \ (feature, lable) 其中 feature 是 pcm, lable 是 one-hot 

    11> batch py> next(pop())[0]   \ 這是 64 組聲音 features
    Looking for data spoken_numbers_pcm.tar in data/
    Extracting data/spoken_numbers_pcm.tar to data/
    'tar' is not recognized as an internal or external command,
    operable program or batch file.
    Data ready!
    loaded batch of 2402 files

    11> .s
          0: [array([[  70.0763561 ,   63.62473879,   38.23389122, ...,    0.        ,
               0.        ,    0.        ],
           [ 177.5696963 ,  205.61660553,  239.15682288, ...,    0.        ,
               0.        ,    0.        ],
           [-106.36758569, -129.81046683, -148.66351426, ...,    0.        ,
               0.        ,    0.        ],
           ...,
           [  -3.62526335,   -6.07374993,   -8.44170319, ...,    0.        ,
               0.        ,    0.        ],
           [  13.55632784,   10.96203872,   11.74544544, ...,    0.        ,
               0.        ,    0.        ],
           [  -1.35571805,   -7.27423625,  -11.18825843, ...,    0.        ,
               0.        ,    0.        ]]), array([[  66.69145693,   61.06051069,   42.44057717, ...,    0.        ,
               0.        ,    0.        ]])] (<class 'list'>)
    11> depth . cr ==> 1
    11> constant next(batch)[0]
    
    11> next(batch)[0] type . cr ==> <class 'list'>
    11> next(batch)[0] dir . cr ==> ['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
    11> next(batch)[0] count . cr ==> 64 <-- batch_size
    
    11> batch py> next(pop()) constant next(batch)
    11> next(batch) type . cr ==> <class 'tuple'> 這 tuple 是 (pcm[64],label[64])
    11> next(batch) count . cr ==> 2 <-- pcm and lable
    11> next(batch) :> [1] . cr
        [array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), 
         array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), 
         array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]), 
         ... snip ...
         array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])]
    11>
    11> next(batch)[0] :> [0] . cr
    [[  70.0763561    63.62473879   38.23389122 ...,    0.            0.            0.        ]
     [ 177.5696963   205.61660553  239.15682288 ...,    0.            0.            0.        ]
     [-106.36758569 -129.81046683 -148.66351426 ...,    0.            0.            0.        ]
     ...,
     [  -3.62526335   -6.07374993   -8.44170319 ...,    0.            0.            0.        ]
     [  13.55632784   10.96203872   11.74544544 ...,    0.            0.            0.        ]
     [  -1.35571805   -7.27423625  -11.18825843 ...,    0.            0.            0.        ]]
     
    不知道這 mfcc 是啥，只知道是聲音檔 
    11> next(batch)[0] :> [0] count . cr ==>  20   \ width = 20  # mfcc features
    11> next(batch)[0] :> [0][0] count . cr ==> 80 \ height = 80  # (max) length of utterance 
    11> next(batch)[0] :> [0][0] . cr
    [  70.0763561    63.62473879   38.23389122   33.16041049   -0.38796544
      -20.28952868  -20.41362055  -18.99775059  -15.50313895  -11.91663208
      -11.14165283  -25.10749751  -23.12044151   -8.37803205   99.59400154
      120.31006276    0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.            0.
        0.            0.            0.            0.            0.        ]
    11>

    [ ] 為何語音 data 是這個擺放結構？
        width = 20  # mfcc features
        height = 80  # (max) length of utterance 
        
    \ 查看 nn 的 save restore 怎麼用
    11> model :> load py: help(pop())
    Help on method load in module tflearn.models.dnn:

    load(model_file, weights_only=False, **optargs) method of tflearn.models.dnn.DNN instance
        Load.

        Restore model weights.

        Arguments:
            model_file: `str`. Model path.
            weights_only: `bool`. If True, only weights will be restored (
                and not intermediate variable, such as step counter, moving
                averages...). Note that if you are using batch normalization,
                averages will not be restored as well.
            optargs: optional extra arguments for trainer.restore (see helpers/trainer.py)
                     These optional arguments may be used to limit the scope of
                     variables restored, and to control whether a new session is
                     created for the restored variables.

    11> model :> save py: help(pop())
    Help on method save in module tflearn.models.dnn:

    save(model_file) method of tflearn.models.dnn.DNN instance
        Save.

        Save model weights.

        Arguments:
            model_file: `str`. Model path.

    11>

    [ ] 搞懂 NN 的 save-restore 它好像是不肯 overwrite 現有檔名
        OK <accept> <py>
        for i in range(1,10000):
            if i % 1000 == 0:
                print("saved_networks/tflearn.lstm.model."+str(i))
        </py> </accept> dictate
        saved_networks/tflearn.lstm.model.1000
        saved_networks/tflearn.lstm.model.2000
        saved_networks/tflearn.lstm.model.3000
        saved_networks/tflearn.lstm.model.4000
        saved_networks/tflearn.lstm.model.5000
        saved_networks/tflearn.lstm.model.6000
        saved_networks/tflearn.lstm.model.7000
        saved_networks/tflearn.lstm.model.8000
        saved_networks/tflearn.lstm.model.9000
        OK
        OK
        OK

    \ 想查出目前的 training steps 是多少。因為 model.load() 進來已經成功，導致
      這個 for loop 的起點不一定是 1 了，要改成 model.load() 進來的當時 training
      step 才對。
      
        for i in range(1,int(training_iters/epoch_count)):
    
    [x] 意外發現 神經網路內部相關的東西
        22> model :> get_train_vars() . cr
        [
        <tf.Variable 'LSTM/LSTM/BasicLSTMCell/Linear/Matrix:0' shape=(208, 512) dtype=float32_ref>
        , 
        <tf.Variable 'LSTM/LSTM/BasicLSTMCell/Linear/Bias:0' shape=(512,) dtype=float32_ref>
        , 
        <tf.Variable 'FullyConnected/W:0' shape=(128, 10) dtype=float32_ref>
        , 
        <tf.Variable 'FullyConnected/b:0' shape=(10,) dtype=float32_ref>
        ]
    22>
        
'''
