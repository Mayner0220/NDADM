import tensorflow as tf
from train import OPTIMIZER, LEARNING_RATE
from dataset import TRAIN_DATASET, TEST_DATASET

EPOCH = 500
BATCH_SIZE = 3

sess = tf.compat.v1.Session()
sess.run(tf.glorot_normal_initializer)

print("Start Learning")
# for epoch in range(EPOCH):