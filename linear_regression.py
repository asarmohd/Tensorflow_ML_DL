# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:45:09 2018

@author: ma
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

x_data = np.linspace(0,10,100000)
noise = np.random.rand(len(x_data))
#y = mx+c
y_true  = (0.25* x_data) + 5 +noise

final_data = pd.concat([pd.DataFrame(data = x_data,columns=['X']),pd.DataFrame(data = y_true,columns=['Y'])],axis=1)

final_data.head()

graph = final_data.sample(200)
graph.plot(kind='scatter',x='X',y='Y')

######## TENSOR FLOW
batch_size   = 10
m = tf.Variable(0.18)
b = tf.Variable(3.9)

x_batch = tf.placeholder(tf.float32,[batch_size])
y_batch = tf.placeholder(tf.float32,[batch_size])

y_model = m*x_batch + b

#Loss Funtion

error = tf.reduce_sum(tf.square(y_batch-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 100000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data),size = batch_size)
        feed = {x_batch:x_data[rand_ind],y_batch:y_true[rand_ind]}
        sess.run(train,feed_dict = feed)
    model_m,model_b = sess.run([m,b])
    
model_m
model_b

y_hat = x_data * model_m + model_b

graph.plot(kind='scatter',x='X',y='Y')
plt.plot(x_data,y_hat,'r')