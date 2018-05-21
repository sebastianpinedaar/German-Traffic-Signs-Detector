# -*- coding: utf-8 -*-

import numpy as np
import cv2
from os.path import isfile, join
from os import listdir, makedirs
from scipy.misc import imread
from sklearn.externals import joblib
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

def train_model2 (train_images_path):
    model_path = join("models", "model2")

    folders = [f for f in listdir(train_images_path) if not(isfile(join(train_images_path, f)))]
    classes =  folders
    
    x_train= np.zeros([1,16*16*3])
    
    y_train= np.zeros([1])
    
    for single_class in classes:
        folder = join(train_images_path, single_class)
        imgs = [f for f in listdir(folder) if isfile(join(folder, f))]
         
        for img in imgs:
            
            img_matrix = imread(join(folder,img))
            res = cv2.resize(img_matrix, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
            x_train = np.vstack([x_train, res.reshape(-1)])
            y_train = np.vstack([y_train, classes.index(single_class) ])
          
    x_train = x_train[1:, :]
    y_train = y_train[1:]
    
    oneHot = OneHotEncoder()
    oneHot.fit(y_train)
                
    y_train_oh = oneHot.transform(y_train).toarray().astype('float32')
    x_train = x_train.astype('float32')
        
    scaler=preprocessing.StandardScaler().fit(x_train)
    
    x_train_scaled = scaler.transform(x_train)
   
    joblib.dump(scaler, join('models', 'model2',"saved",'scaler.pkl') )
    joblib.dump(oneHot, join('models', 'model2',"saved",'onehot.pkl'))
        
    num_epochs=500
    learning_rate=0.1
    
    X = tf.placeholder(tf.float32, [None, 16*16*3], name="input")
    y = tf.placeholder(tf.float32, [None, 43], "input_label")
    
    W = tf.Variable(tf.zeros([16*16*3,43]))
    b = tf.Variable(tf.zeros([43]))
    
    y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b), name="softmax_output")
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_), reduction_indices=1))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for echo in range(num_epochs):
            _, c = sess.run([optimizer, cost], feed_dict={X:x_train_scaled, y:y_train_oh})
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            print("Epoch:"+str(echo))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy_measure")
            #print("Accuracy:", accuracy.eval({X: x_train_scaled, y: y_train_oh}))

        final_accuracy = accuracy.eval({X: x_train_scaled, y: y_train_oh})        
        saver.save(sess, join(model_path,"saved","model2.ckpt"))
            

    print("Train error: "+str(1-final_accuracy))
    


def test_model2 (test_images_path):
 
    folders = [f for f in listdir(test_images_path) if not(isfile(join(test_images_path, f)))]
    classes =  folders
    
    x_test= np.zeros([1,16*16*3])
    y_test= np.zeros([1])
    
    for single_class in classes:
        folder = join(test_images_path, single_class)
        imgs = [f for f in listdir(folder) if isfile(join(folder, f))]
         
        for img in imgs:
            
            img_matrix = imread(join(folder,img))
            res = cv2.resize(img_matrix, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
            x_test = np.vstack([x_test, res.reshape(-1)])
            y_test = np.vstack([y_test, classes.index(single_class) ])
          
    x_test = x_test[1:, :]
    y_test = y_test[1:]
    
    oneHot = joblib.load(join('models','model2',"saved",'onehot.pkl'))
    y_test_oh = oneHot.transform(y_test).toarray().astype('float32')
    x_test = x_test.astype('float32')
        
    
    scaler = joblib.load(join('models','model2',"saved",'scaler.pkl'))
    x_test_scaled = scaler.transform(x_test)

    saver = tf.train.import_meta_graph(join('models', 'model2', "saved",'model2.ckpt.meta'))
    graph = tf.get_default_graph()
    X= graph.get_tensor_by_name("input:0")
    y = graph.get_tensor_by_name("input_label:0")
    accuracy = graph.get_tensor_by_name("accuracy_measure:0")
    with tf.Session() as sess:

      saver.restore(sess, join("models","model2", "saved","model2.ckpt"))
      test_accuracy = accuracy.eval({X: x_test_scaled, y: y_test_oh})
        # cv2.imshow(img_matrix)
    print("Test error: "+str(1-test_accuracy))
