# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:13:23 2019

@author: wgh
"""

import tensorflow as tf


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def next_batch():
    
    filename_queue = tf.train.string_input_producer(["C://Users//wgh//Desktop//Oxford//jpg//oxfordpicture.tfrecords"])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                   features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw' : tf.FixedLenFeature([], tf.string),
                                    })#return image and label
    
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 3])  #reshape image to 500*500*3
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32) #throw label tensor
    batch_size = 80
    capacity = 10000
    example_batch,label_batch = tf.train.shuffle_batch([img,label],batch_size = batch_size,capacity=capacity,min_after_dequeue=1280)
    return example_batch,label_batch

x_image=tf.placeholder(tf.float32,[80,28,28,3])
y_previous=tf.placeholder(tf.int32,[80])
y_=tf.one_hot(y_previous,17)#one_hot稀疏编码

W_conv1=weight_variable([5,5,3,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)#14*14*32

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)#7*7*64

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[80,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,17])
b_fc2=bias_variable([17])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
#cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
trainstep = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
if __name__ == "__main__":
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_local_variables())
    data_batchs = next_batch()
    coord = tf.train.Coordinator()
    iterations=0
    threads = tf.train.start_queue_runners(sess,coord)
    
    try:
        while not coord.should_stop():
            data = sess.run(data_batchs)
            iterations+=1
            if(iterations==100):#iterations设置迭代的训练次数
                break
            print(iterations,data[0],data[1])
            for i in range(10):
                sess.run(trainstep,feed_dict={x_image:data[0],y_previous:data[1],keep_prob:0.5})                      
                print("step %d,trainning accuracy %g" %(i,sess.run(accuracy,feed_dict={x_image:data[0],y_previous:data[1],keep_prob:1.0})))
    except tf.errors.OutOfRangeError:
        print("complete")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()