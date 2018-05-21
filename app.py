# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:24:11 2018

@author: User
"""
import click
import urllib
import zipfile
import matplotlib.pyplot as plt
from download import download, train_test_folders
from scipy.misc import imread
import numpy as np
import cv2
from os.path import isfile, join
from os import listdir, makedirs
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import click
from sklearn.externals import joblib
from model1 import train_model1, test_model1
from model2 import train_model2, test_model2
from model3 import train_model3, test_model3
from imshow_labels import imshow_labels

@click.command()
@click.argument('command')
@click.option('-m', default="model1", help='Model to execute. POssible models are: logistic regression using sk-learn (model1), \
                                                          logistic regression in tensorflow (model2) and lenet5 in tensor flow (model3')
@click.option('-d', help="location of data to use")

def main(command, m, d):
    
    if (command=="download"):
        
        print("Descargando datos...")
        download()
        train_test_folders()
    if (command=="train"):
        
        if(m=="model1"):
            train_model1(d)
            
        elif(m=="model2"):
            train_model2(d)
            
        else:
            train_model3(d)
    
    if(command =="test"):
        
        if(m=="model1"):
            test_model1(d)
            
        elif(m=="model2"):
            test_model2(d)
            
        else:
            test_model3(d)
            
    if (command =="infer"):
        
        if(m=="model1"):
            print("Infering with model1")
            clf = joblib.load(join('models','model1',"saved",'model1.pkl'))
            
            imgs= [f for f in listdir(d) if isfile(join(d, f))]
            x_train= np.zeros([1,16*16*3])
            img_to_read = []
            
            for img in imgs:
                img_to_read.append(join(d, img))
                img_matrix = imread(join(d,img))
                res = cv2.resize(img_matrix, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
                x_train = np.vstack([x_train, res.reshape(-1)])
            
            x_train= x_train[1:,:]
            labels = clf.predict(x_train)
            
            imshow_labels(img_to_read, labels)
            print(clf.predict(x_train))
            
               
        elif(m=="model2"):
            print("Infering with model2")

            imgs= [f for f in listdir(d) if isfile(join(d, f))]
            print(imgs)
            x_train= np.zeros([1,16*16*3])
            img_to_read=[]
            
            for img in imgs:
                img_to_read.append(join(d, img))
                img_matrix = imread(join(d,img))
                res = cv2.resize(img_matrix, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
                x_train = np.vstack([x_train, res.reshape(-1)])      
            
            scaler = joblib.load(join('models','model2',"saved",'scaler.pkl'))
            x_train= x_train[1:,:]

            x_train_scaled = scaler.transform(x_train)
            print(x_train_scaled.shape)
            saver = tf.train.import_meta_graph(join('models', 'model2', "saved",'model2.ckpt.meta'))
            graph = tf.get_default_graph()
            y_= graph.get_tensor_by_name("softmax_output:0")
            X= graph.get_tensor_by_name("input:0")
            
            with tf.Session() as sess:
              
              saver.restore(sess, join("models","model2", "saved","model2.ckpt"))
              y1= sess.run(y_, feed_dict={X:x_train_scaled})
                
            labels=np.argmax(y1, axis=1)
            imshow_labels(img_to_read, labels.tolist())
            
        elif(m=="model3"):
            print("Infering with model3")
            
            imgs= [f for f in listdir(d) if (isfile(join(d, f)))]
                
            x_test= np.zeros([1,32,32,3])

            img_to_read=[]

            for img in imgs:
                
                img_to_read.append(join(d, img))
                img_matrix = imread(join(d,img))
                
                res = cv2.resize(img_matrix, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
                print(res.shape)
                x_test[x_test.shape[0]-1,:,:,:] = res
                x_test= np.vstack([x_test, np.zeros([1,32,32,3])])

        
            saver = tf.train.import_meta_graph(join('models', 'model3',"saved", 'model3.ckpt.meta'))
            graph = tf.get_default_graph()
            X= graph.get_tensor_by_name("input:0")
            y= graph.get_tensor_by_name("softmax_output:0")

            x_test = x_test.astype('float32')
            
            x_test_n = x_test/255
 
            with tf.Session() as sess:
              saver.restore(sess, join("models","model3","saved", "model3.ckpt"))
              l= y.eval({X: x_test_n})
            labels = np.argmax(l, axis=1)
            imshow_labels(img_to_read, labels.tolist())
          
            
if __name__ == '__main__':
    main()

