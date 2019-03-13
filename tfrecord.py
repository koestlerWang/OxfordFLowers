# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:45:47 2019

@author: wgh
"""

import os
import tensorflow as tf


path="C://Users//wgh//Desktop//Oxford//jpg//"
classes=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
#classes=['0','1']
writer_train = tf.python_io.TFRecordWriter("C://Users//wgh//Desktop//Oxford//jpg//train.tfrecords")#将每个图片集的前六十张图片作为训练集
writer_test  = tf.python_io.TFRecordWriter("C://Users//wgh//Desktop//Oxford//jpg//test.tfrecords")#将每个图片集的后二十张图片作为测试集

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  
with tf.Session() as sess:
    for index,name in enumerate(classes):
        class_path=path+name+'//'
        iterations=0
        for imagename in os.listdir(class_path):
            iterations=iterations+1
            if(iterations<=60):
                imag_path=class_path+imagename
                image = tf.read_file(imag_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image_raw=(image.eval()).tostring()
                example=tf.train.Example(features=tf.train.Features(feature={
                            'label':_int64_feature(index),
                            'image_raw':_bytes_feature(image_raw)
                            }
                    ))
                writer_train.write(example.SerializeToString())
            elif(iterations>60):
                imag_path=class_path+imagename
                image = tf.read_file(imag_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image_raw=(image.eval()).tostring()
                example=tf.train.Example(features=tf.train.Features(feature={
                            'label':_int64_feature(index),
                            'image_raw':_bytes_feature(image_raw)
                            }
                    ))
                writer_test.write(example.SerializeToString())
    writer_train.close()
    writer_test.close()
    
#"C://Users//wgh//Desktop//Oxford//jpg//output.tfrecords"

