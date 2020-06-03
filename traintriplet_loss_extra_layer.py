# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:57:42 2020

@author: vikas
"""

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras.models import Sequential
from data_generator_2 import *
from converter.model import ScaleLayer, ReshapeLayer
ALPHA = 0.2

def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

PATH_TO_DLIB_H5_MODEL = 'model/dlib_face_recognition_resnet_model_v1.h5'
# load model
model_from_file = tf.keras.models.load_model(PATH_TO_DLIB_H5_MODEL, custom_objects={'ScaleLayer': ScaleLayer, 'ReshapeLayer': ReshapeLayer})
    
top_model = Sequential()
top_model.add(Dense(128, input_shape=(128,), use_bias = False))

model = Model(inputs=model_from_file.input, outputs=top_model(model_from_file.output))
model.summary()

for layer in model.layers[:-1]:
    layer.trainable = False

# for layer in model.layers:
#     layer.trainable = True


IMAGE_SIZE = 150
NUM_EPOCHS = 30
STEPS_PER_EPOCH = 10
input_shape=( IMAGE_SIZE, IMAGE_SIZE, 3)
A = Input(shape=input_shape, name = 'anchor')
P = Input(shape=input_shape, name = 'anchorPositive')
N = Input(shape=input_shape, name = 'anchorNegative')

enc_A = model(A)
enc_P = model(P)
enc_N = model(N)

opt = SGD(lr=0.0001, momentum=0.9)

# opt = Adam(learning_rate=0.0001)

tripletModel = Model(inputs=[A, P, N], outputs=[enc_A, enc_P, enc_N])
tripletModel.compile( optimizer = opt, loss = triplet_loss)

gen = batch_generator(10)

# Callbacks
# early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)
STAMP = 'facenet_10' 
# checkpoint_dir = './' + 'checkpoints/2905/' + str(int(time.time())) + '/'
checkpoint_dir = './' + 'checkpoints/3105/' + '/'

bst_model_path = checkpoint_dir + STAMP + '.h5'

# tensorboard = Tensorboard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))
filepath="./checkpoints/smooth_L2-{epoch}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='model_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
tripletModel.fit_generator(gen, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,  callbacks=callbacks_list)
# tripletModel.save('checkpoints/my_model') 
model.save(bst_model_path)
