# -*- coding: utf-8 -*-

import numpy as np
import cv2
from os.path import isfile, join
from os import listdir, makedirs
from scipy.misc import imread
from sklearn.externals import joblib

def train_model1(train_images_path):

    
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
      
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression(C=1., solver='lbfgs', multi_class='multinomial')
    lr.fit(x_train, y_train)
    y_train_pred = lr.predict(x_train)
    
    
    train_error = np.mean(np.transpose(y_train)!=y_train_pred)
    print("Train error: "+str(train_error))
    
    from sklearn.externals import joblib
    joblib.dump(lr, join('models', 'model1',"saved",'model1.pkl') )


def test_model1(test_images_path):
   
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
    
    from sklearn.linear_model import LogisticRegression
      
    clf = joblib.load(join('models','model1',"saved",'model1.pkl'))
    y_test_pred = clf.predict(x_test)
    test_error = np.mean(np.transpose(y_test)!=y_test_pred)
    print("Test error: "+str(test_error))