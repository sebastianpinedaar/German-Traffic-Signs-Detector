# -*- coding: utf-8 -*-
import cv2
import numpy as np

def imshow_labels (images, labels):
    
    
    for label, img_name in zip(labels, images):
          
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        img = cv2.imread(img_name)
        img = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)

        cv2.putText(img,"Label: "+str(label), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        cv2.imshow("img",img)
        
        cv2.waitKey(0)
        #Display the image
        

        
    return 0
