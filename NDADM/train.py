import tensorflow as tf
import numpy as np
from dataset import TRAIN_DATASET, TEST_DATASET

LEARNING_RATE = 0.01


X_train = tf.compat.v1.placeholder(tf.float32, [None, 36608])
X_img = tf.reshape(X_train, [-1, 176, 208, 1])

Y_train = tf.compat.v1.placeholder(tf.float32, [None, 4])

W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.1))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
L1a = tf.nn.relu(L1)
L1m = tf.nn.max_pool2d(L1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.1))
L2 = tf.nn.conv2d(L1m, W2, strides=[1, 1, 1, 1], padding="SAME")
L2a = tf.nn.relu(L2)
L2m = tf.nn.max_pool2d(L2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W3 = tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.1))
L3 = tf.nn.conv2d(L2m, W3, strides=[1, 1, 1, 1], padding="SAME")
L3a = tf.nn.relu(L3)
L3m = tf.nn.max_pool2d(L3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W4 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.1))
L4 = tf.nn.conv2d(L3m, W4, strides=[1, 1, 1, 1], padding="SAME")
L4a = tf.nn.relu(L4)
L4m = tf.nn.max_pool2d(L4a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

W5 = tf.Variable(tf.random.normal([3, 3, 16, 32], stddev=0.1))
L5 = tf.nn.conv2d(L4m, W5, strides=[1, 1, 1, 1], padding="SAME")
L5a = tf.nn.relu(L5)
L5m = tf.nn.max_pool2d(L5a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

L_F = tf.reshape(L5m, [-1, 6*7*32])