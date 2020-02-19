# -*- coding: utf-8 -*-
"""LSTMforall.ipynb
author: Sikai Yin
"""

import pandas as pd
import numpy as np

drive.mount('/content/drive')
df_test = pd.read_csv('./testing.csv')
alist = list(df_test.groupby('Stock Code').groups.keys())

def lstm_seq2seq(input):   #LSTM Seq2Seq with attention model
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from datetime import datetime
    from datetime import timedelta
    sns.set()

    df_train = pd.read_csv('/content/drive/My Drive/wholeday/[' + str(input) +'].csv')
    date_ori = pd.to_datetime(df.iloc[:, 1]).tolist()

    minmax = MinMaxScaler().fit(df_train.iloc[:, 3:].astype('float32'))
    df_log = minmax.transform(df_train.iloc[:, 3:].astype('float32'))
    df_log = pd.DataFrame(df_log)

    num_layers = 1
    size_layer = 128
    timestamp = 5
    epoch = 100
    dropout_rate = 0.7
    future_day = 25
    
    class Model:
    
        def __init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias = 0.1,
                    attention_size = 100):
            def lstm_cell():
                return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)
            self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)],
                                                         state_is_tuple = False)
            self.X = tf.placeholder(tf.float32, [None, None, size])
            self.Y = tf.placeholder(tf.float32, [None, output_size])
            self.hidden_layer = tf.placeholder(tf.float32, (None, num_layers * 2 * size_layer))
            drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob = forget_bias)
            self.outputs, last_state = tf.nn.dynamic_rnn(drop, self.X, 
                                                   initial_state = self.hidden_layer,
                                                   dtype = tf.float32)
            attention_w = tf.get_variable("attention_v", [attention_size], tf.float32)
            query = tf.layers.dense(tf.expand_dims(last_state[:,size_layer:], 1), attention_size)
            keys = tf.layers.dense(self.outputs, attention_size)
            align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
            align = tf.nn.tanh(align)
            self.outputs = tf.squeeze(tf.matmul(tf.transpose(self.outputs, [0, 2, 1]),
                                                 tf.expand_dims(align, 2)), 2)
            self.outputs = tf.concat([self.outputs,last_state[:,size_layer:]],1)
            with tf.variable_scope("decoder", reuse = False):
                self.rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)],
                                                                state_is_tuple = False)
                drop_dec = tf.contrib.rnn.DropoutWrapper(self.rnn_cells_dec, output_keep_prob = forget_bias)
                self.outputs_dec, self.last_state = tf.nn.dynamic_rnn(drop_dec, self.X, 
                                                       initial_state = self.outputs,
                                                       dtype = tf.float32)
            query_dec = tf.layers.dense(tf.expand_dims(self.last_state[:,size_layer:], 1), attention_size)
            keys_dec = tf.layers.dense(self.outputs_dec, attention_size)
            align_dec = tf.reduce_sum(attention_w * tf.tanh(keys_dec + query_dec), [2])
            align_dec = tf.nn.tanh(align_dec)
            self.outputs_dec = tf.squeeze(tf.matmul(tf.transpose(self.outputs_dec, [0, 2, 1]),
                                                 tf.expand_dims(align_dec, 2)), 2)
            self.rnn_W = tf.Variable(tf.random_normal((size_layer, output_size)))
            self.rnn_B = tf.Variable(tf.random_normal([output_size]))
            self.logits = tf.matmul(self.outputs_dec, self.rnn_W) + self.rnn_B
            self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
            
        tf.reset_default_graph()
    modelnn = Model(0.001, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        init_value = np.zeros((timestamp, num_layers * 2 * size_layer))
        total_loss = 0
        for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
            batch_x = np.expand_dims(df_log.iloc[k: k + timestamp, :].values, axis = 1)
            batch_y = df_log.iloc[k + 1: k + timestamp + 1, :].values
            last_state, _, loss = sess.run([modelnn.last_state, 
                                            modelnn.optimizer, 
                                            modelnn.cost], feed_dict={modelnn.X: batch_x, 
                                                                      modelnn.Y: batch_y, 
                                                                      modelnn.hidden_layer: init_value})
            init_value = last_state
            total_loss += loss
        total_loss /= (df_log.shape[0] // timestamp)
        #print('epoch:', i + 1, 'avg loss:', total_loss)
        
    output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
    output_predict[0, :] = df_log.iloc[0, :]
    upper_b = (df_log.shape[0] // timestamp) * timestamp
    init_value = np.zeros((timestamp, num_layers * 2 * size_layer))
    for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[k: k + timestamp, :], axis = 1),
                                         modelnn.hidden_layer: init_value})
        init_value = last_state
        output_predict[k + 1: k + timestamp + 1, :] = out_logits

    expanded = np.expand_dims(df_log.iloc[upper_b: , :], axis = 1)
    out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:expanded,
                                         modelnn.hidden_layer: init_value[-expanded.shape[0]:]})
    init_value[-expanded.shape[0]:] = last_state
    output_predict[upper_b + 1: df_log.shape[0] + 1, :] = out_logits
    df_log.loc[df_log.shape[0]] = out_logits[-1, :]
    
    
    for i in range(future_day - 1):
        out_logits, last_state = sess.run([modelnn.logits, modelnn.last_state], feed_dict = {modelnn.X:np.expand_dims(df_log.iloc[-timestamp:, :], axis = 1),
                                         modelnn.hidden_layer: init_value})
        init_value = last_state
        output_predict[df_log.shape[0], :] = out_logits[-1, :]
        df_log.loc[df_log.shape[0]] = out_logits[-1, :]
        date_ori.append(date_ori[-1]+timedelta(days=1))
    
    df_log = minmax.inverse_transform(output_predict)
    df_loglog = pd.DataFrame(df_log)
    result = df_loglog.iloc[-30:, 4]
    return result

data = np.zeros((31,len(alist)))
for i in range(len(alist)):
    data[0][i] = alist[i]
    try:
        data[1:,i] = lstm_seq2seq(alist[i])
        print(data[1:,i])

    except:
        data[1:,i] = 0
        continue

results = pd.DataFrame(data=data[1:], index = None, columns=data[0])
results.to_csv('result1.csv')

