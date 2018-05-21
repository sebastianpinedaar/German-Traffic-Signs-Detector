# -*- coding: utf-8 -*-

import numpy as np
import cv2
from os.path import isfile, join
from os import listdir, makedirs
from scipy.misc import imread
import tensorflow as tf
import tensorflow.contrib.layers
from tensorflow.contrib.layers import flatten
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

def train_model3(train_images_path):
    
    model_path = join("models", "model3")
    folders = [f for f in listdir(train_images_path) if not(isfile(join(train_images_path, f)))]
    classes =  folders
        
    x_train= np.zeros([1,32,32,3])
    
    y_train= np.zeros([1])
    
    for single_class in classes:
        
        folder = join(train_images_path, single_class)
        imgs = [f for f in listdir(folder) if isfile(join(folder, f))]
         
        for img in imgs:
            
            img_matrix = imread(join(folder,img))
            res = cv2.resize(img_matrix, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            x_train[x_train.shape[0]-1,:,:,:] = res
            x_train = np.vstack([x_train, np.zeros([1,32,32,3])])
            y_train = np.vstack([y_train, classes.index(single_class) ])
     
    oneHot = OneHotEncoder()
    oneHot.fit(y_train)
    
    # transform
    x_train = x_train.astype('float32')
    
    x_train_n = x_train/255
    
    y_train_oh = oneHot.transform(y_train).toarray().astype('float32')
    joblib.dump(oneHot, join('models', 'model3',"saved",'onehot.pkl') )
    
            
    #Placeholders
    learning_rate=0.1
           
    X = tf.placeholder("float32", [None, 32,32,3], name="input")
    y = tf.placeholder("float32", [None, 43], name="input_label")
    
    tf.set_random_seed(1)
    
    initializer = tf.contrib.layers.xavier_initializer (seed=0)
    
    
    CONV1_W = tf.Variable(initializer((5,5,3,6)))
    CONV1_b = tf.Variable(tf.zeros(6))
    
    CONV2_W = tf.Variable(initializer((5,5,6,16)))
    CONV2_b = tf.Variable( tf.zeros(16))
    
    FC1_W = tf.Variable(initializer((400, 120)))
    FC1_b = tf.Variable(tf.zeros(120))
    
    FC2_W = tf.Variable(initializer((120,84)))
    FC2_b = tf.Variable( tf.zeros(84))
    
    SM_W = tf.Variable(initializer((84,43)))
    SM_b = tf.Variable( tf.zeros(43))
    
    Z1 = tf.nn.conv2d(X,CONV1_W, strides=[1,1,1,1], padding="VALID") + CONV1_b
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    
    Z2 = tf.nn.conv2d(P1, CONV2_W, strides= [1,1,1,1], padding ="VALID") + CONV2_b
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    
    FC0 = flatten(P2)
    
    FC1 = tf.add(tf.matmul(FC0, FC1_W) ,FC1_b)
    FC1_O = tf.nn.relu(FC1)
    
    FC2 = tf.add(tf.matmul(FC1_O, FC2_W) ,FC2_b)
    FC2_O = tf.nn.relu(FC2)
    
    SM = tf.add(tf.matmul(FC2_O, SM_W), SM_b)
    y_ = tf.nn.softmax(SM, name="softmax_output")
   
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
    loss_operation = tf.reduce_mean(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy_measure")
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_operation)
 
    saver = tf.train.Saver()
    
    num_epochs = 100
    
    e_v=[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(num_epochs):
            A,_, e, W= sess.run([y_,optimizer, accuracy_operation, CONV1_W], feed_dict={X: x_train_n, y:y_train_oh })
            print("Epoch:"+str(i))  
            e_v.append(e)
            saver.save(sess, join("models", "model3","saved","model3.ckpt"))
        train_accuracy = accuracy_operation.eval({X: x_train_n, y: y_train_oh})
    print("Train error:"+str(1-train_accuracy))
  
    
def test_model3(test_images_path):
    
    folders = [f for f in listdir(test_images_path) if not(isfile(join(test_images_path, f)))]
    classes =  folders
        
    x_test= np.zeros([1,32,32,3])
    y_test = np.zeros([1])
    
    for single_class in classes:

        folder = join(test_images_path, single_class)
        imgs = [f for f in listdir(folder) if isfile(join(folder, f))]

        for img in imgs:
            
            img_matrix = imread(join(folder,img))
            res = cv2.resize(img_matrix, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            x_test[x_test.shape[0]-1,:,:,:] = res
            x_test= np.vstack([x_test, np.zeros([1,32,32,3])])
            y_test = np.vstack([y_test, classes.index(single_class) ])
     
    
    oneHot = joblib.load(join('models','model3',"saved",'onehot.pkl'))

    saver = tf.train.import_meta_graph(join('models', 'model3', "saved",'model3.ckpt.meta'))
    graph = tf.get_default_graph()
    X= graph.get_tensor_by_name("input:0")
    y= graph.get_tensor_by_name("input_label:0")
    accuracy = graph.get_tensor_by_name("accuracy_measure:0")
    # transform
    x_test = x_test.astype('float32')
    
    x_test_n = x_test/255
    
    y_test_oh = oneHot.transform(y_test).toarray().astype('float32')
    
    
    with tf.Session() as sess:
      saver.restore(sess, join("models","model3", "saved","model3.ckpt"))
      test_accuracy = accuracy.eval({X: x_test_n, y: y_test_oh})
      
    print("Test error: "+str(1-test_accuracy))
    
    
    

