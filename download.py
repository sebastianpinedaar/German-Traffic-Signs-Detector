# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:52:48 2018

@author: User
"""
def download():
    
    import urllib
    import zipfile
    
    url_o = urllib.request.urlretrieve("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip", filename="temp_data")

    zip_ref = zipfile.ZipFile(url_o[0],'r')
    zip_ref.extractall(join("images","user"))
    zip_ref.close()
    
    
def train_test_folders():
    
    
    from os import listdir, makedirs
    from os.path import isfile, join
    import pandas as pd
    from shutil import copyfile
    import random 

        
    all_images_path = join("images", "FullIJCNN2013")
    train_images_path = join("images", "train")
    test_images_path = join("images", "test")
    folders = [f for f in listdir(all_images_path) if not(isfile(join(all_images_path, f)))]
    
    
    for folder in folders:
        
        print("Working on folder: "+folder)
        list_images = pd.Series([f for f in listdir(join(all_images_path,folder)) if isfile(join(all_images_path,folder, f))])
        n_images= len(list_images)
        train_idx = random.sample(range(n_images), int(n_images*0.8))
        train_img = list_images[train_idx]
        test_img = list_images[[not(i in train_idx) for i in range(n_images)]]
        
        makedirs(join(train_images_path, folder), exist_ok=True)
        makedirs(join(test_images_path, folder), exist_ok=True)
    
        for file in train_img:
            copyfile(join(all_images_path,folder, file), join(train_images_path, folder,file))
            
        for file in test_img:
            copyfile(join(all_images_path,folder, file), join(test_images_path, folder,file))
        