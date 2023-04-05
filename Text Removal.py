#!/usr/bin/env python
# coding: utf-8

# ## Keras-OCR text removal test

# In[63]:


#Libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras_ocr
import cv2
import os
import math
import pydicom


# In[16]:


os.getcwd()


# In[25]:


#Display the image
img = "C:\\Users\\Chris\\School\\Test\\Test_text.jpg"
test = plt.imread(img)
plt.imshow(test)


# ### Test Keras-OCR to remove all the texts on the book

# In[32]:


pipeline = keras_ocr.pipeline.Pipeline()

#read in the image with keras-ocr
img_test = keras_ocr.tools.read(img)


prediction_groups = pipeline.recognize([img_test])

#print image with annotations
keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])


# In[34]:


for group in prediction_groups:
    for x in group:
        print(x)


# In[35]:


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


# In[55]:


def inpaint_text(img_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(img_path, img_rgb)
                 
    return(img)


# In[56]:


no_text1 = inpaint_text(img)
plt.imshow(no_text1)


# In[59]:


test2 = "C:\\Users\\Chris\\School\\Test\\Test_text2.jpg"

show_test2 = plt.imread(test2)
plt.imshow(show_test2)


# In[60]:


no_text2 = inpaint_text(test2)
plt.imshow(no_text2)


# In[ ]:




