
# coding: utf-8

# In[16]:


import tensorflow as tf
import cv2
import numpy as np   
import sys
import os


# In[19]:


sess=tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('./mnist_model.meta')
saver.restore(sess, './mnist_model')


# In[20]:


graph = tf.get_default_graph()
keep_prob =  graph.get_tensor_by_name("keep_prob:0")
net_output = graph.get_tensor_by_name("net_output:0")
net_input = graph.get_tensor_by_name("net_input:0")


# In[21]:


#path = "C:\\Users\\Tsofan\\Desktop\\ex1\\numbersToTest"
path = sys.argv[1]

for file in os.listdir(path):
    img_path = path + "\\" + file
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (28, 28))
    img_as_arr = np.array(resized_image, dtype = np.float32)
    img_as_arr.flatten()
    img_as_arr = img_as_arr.reshape([1,784])
    
    res = sess.run(net_output,  
                   feed_dict= {'net_input:0': img_as_arr, 'keep_prob:0': 1.0});
    
    finalRes = np.argmax(res)
    print('picture: {} - result: {}'.format(img_path,finalRes))

