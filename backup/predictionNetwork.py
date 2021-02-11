import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Dropout, LayerNormalization, Activation, Softmax, Concatenate
from tensorflow.keras.optimizers import Adam
import multiprocessing
import time
from threading import Thread
import numpy as np

# class PredictionNetwork(multiprocessing.Process):
class PredictionNetwork():
    def __init__(self, block_size, in_block, out_block, lr=0.001):
        # super(PredictionNetwork, self).__init__()
        self.optimizer = Adam(lr=lr)
        self.MSE = tf.keras.losses.MeanSquaredError()
        self.model = self.create_model(input_shape=(block_size, 1))
        self.in_block = in_block
        self.out_block = out_block

    def create_model(self, input_shape, kernel_size=5):
        model = Sequential()
        model.add(Conv1D(400, kernel_size, padding='same', activation='tanh', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(200, kernel_size, padding='same', activation='tanh'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(50, kernel_size, padding='same', activation='tanh'))
        model.add(UpSampling1D(size=2))
        model.add(Conv1D(200, kernel_size, padding='same', activation='tanh'))
        model.add(UpSampling1D(size=2))
        model.add(Conv1D(400, kernel_size, padding='same', activation='tanh'))
        model.add(Conv1D(1, kernel_size, padding='same', activation='tanh'))
        print(model.summary())
        return model

    def train_step(self, model_input, normalize=False):
        # y_true = tf.random.normal(model_input.shape)
        model_input = tf.expand_dims(model_input, 0)
        model_input = tf.expand_dims(model_input, -1)
        y_true = model_input # regression loss on reconstruction
        if normalize:
            y_true *= 1/(tf.math.reduce_max(y_true) + 1e-7) # lazy normalize
            model_input *= 1/(tf.math.reduce_max(model_input) + 1e-7)

        with tf.GradientTape() as tape:
            y_pred = self.model(model_input, training=True)
            loss = self.MSE(y_true, y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # apply gradients to optimizer

        print(f'loss: {loss}')
        return tf.squeeze(y_pred)

    def start_thread(self):
        runThread = Thread(target=self.run, daemon=True)
        runThread.start()

    def run(self):
        running = True
        while running:
            time.sleep(0.001)
            self.out_block[:] = self.train_step(np.asarray(self.in_block), normalize=False)
        