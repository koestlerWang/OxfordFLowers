# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:16:37 2019

@author: wgh
"""

import os
import tensorflow as tf


path="C://Users//wgh//Desktop//Oxford//jpg//"
classes=['12','13','14','15','16']

with tf.Session() as sess:
    for index,name in enumerate(classes):
        class_path=path+name+'//'
        for imagename in os.listdir(class_path):
            imag_path=class_path+imagename
            
            img_raw=tf.gfile.GFile(imag_path,'rb').read()
             #print(type(img_raw))
            img=tf.image.decode_jpeg(img_raw)
            #print(type(img))
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            resized=tf.image.resize_images(img,[500,500],method=0)
            #print(type(resized))
            # 转换图像的数据类型
            img_data = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
            encoded_image=tf.image.encode_jpeg(img_data)
            with tf.gfile.GFile(imag_path,"wb") as f:
                f.write(encoded_image.eval())
                '''
with tf.Session() as sess:
    imag_raw=tf.gfile.GFile("C://Users//wgh//Desktop//Oxford//test//0//image_0006.jpg",'rb').read()
    img=tf.image.decode_jpeg(imag_raw)
    img_data = tf.image.convert_image_dtype(img, dtype=tf.float32)
    resized=tf.image.resize_images(img_data,[200,200],method=1)
    img_data = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
    encoded_image=tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("C://Users//wgh//Desktop//Oxford//test//0//test.jpg","wb") as f:
        f.write(encoded_image.eval())'''