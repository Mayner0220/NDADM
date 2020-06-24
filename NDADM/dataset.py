import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

TRAIN_DATASET = train.flow_from_directory("./Alzheimer_s Dataset/train",
                                          target_size=(208, 176),
                                          batch_size=3,
                                          class_mode='categorical')
TEST_DATASET = test.flow_from_directory("./Alzheimer_s Dataset/test",
                                        target_size=(208, 176),
                                        batch_size=3,
                                        class_mode='categorical')