from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Dropout

def get_alexnet():
    alexnet_model = Sequential()
    alexnet_model.add(Convolution2D(filters=96, kernel_size=11, strides=4,input_shape=(227,227,3), padding='valid', activation='relu'))
    alexnet_model.add(MaxPooling2D(pool_size = 3, strides=2))
    alexnet_model.add(Convolution2D(filters=256, kernel_size=5, padding='same', activation='relu'))
    alexnet_model.add(MaxPooling2D(pool_size = 3, strides=2))
    alexnet_model.add(Convolution2D(filters=384, kernel_size=3, padding='same', activation='relu'))
    alexnet_model.add(Convolution2D(filters=384, kernel_size=3, padding='same', activation='relu'))
    alexnet_model.add(Convolution2D(filters=256, strides=2, kernel_size=3, padding='valid', activation='relu'))
    alexnet_model.add(Flatten())
    alexnet_model.add(Dense(4096, activation='relu'))
    alexnet_model.add(Dropout(0.5))
    alexnet_model.add(Dense(4096, activation='relu'))
    alexnet_model.add(Dropout(0.5))
    alexnet_model.add(Dense(1000, activation='softmax'))
    return alexnet_model

def get_YOLO():
    YOLO_model = Sequential()
    YOLO_model.add(Convolution2D(filters=16, kernel_size=3, strides=1,input_shape=(416,416,3), padding='same', activation='relu'))
    YOLO_model.add(MaxPooling2D(pool_size = 2, strides=2))
    YOLO_model.add(Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    YOLO_model.add(MaxPooling2D(pool_size = 2, strides=2))
    YOLO_model.add(Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    YOLO_model.add(MaxPooling2D(pool_size = 2, strides=2))
    YOLO_model.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    YOLO_model.add(MaxPooling2D(pool_size = 2, strides=2))
    YOLO_model.add(Convolution2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    YOLO_model.add(MaxPooling2D(pool_size = 2, strides=2))
    YOLO_model.add(Convolution2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    YOLO_model.add(MaxPooling2D(pool_size = 2, strides=1, padding='same'))
    YOLO_model.add(Convolution2D(filters=1024, kernel_size=3, padding='same', activation='relu'))
    YOLO_model.add(Convolution2D(filters=1024, kernel_size=3, padding='same', activation='relu'))
    YOLO_model.add(Convolution2D(filters=125, kernel_size=1, padding='same', activation='relu'))
    return YOLO_model
