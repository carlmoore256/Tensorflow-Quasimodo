import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv1DTranspose, AveragePooling1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Dropout, LayerNormalization, Activation, Softmax, Concatenate, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import multiprocessing
import time
from threading import Thread
import numpy as np

# class PredictionNetwork(multiprocessing.Process):
class PredictionNetwork():
    def __init__(self, block_size, in_block, out_block, lock, lr=0.001):
        # super(PredictionNetwork, self).__init__()
        # self.optimizer = Adam(lr=lr)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        self.MSE = tf.keras.losses.MeanAbsoluteError()
        self.model = self.create_model(input_shape=(block_size, 1), kernel_size=9)
        self.in_block = in_block
        self.out_block = out_block
        self.lock = lock
        self.last_block = np.zeros((block_size,))
        self.encoding_dim = 32

    def create_model(self, input_shape, kernel_size=5):
        input_layer = Input((input_shape,))

        encoded = Dense(self.encoding_dim, activation='tanh')(input_layer)
        return model


    def train_step(self, model_input, normalize=False):
        # y_true = tf.random.normal(model_input.shape)
        model_input = tf.expand_dims(model_input, 0)
        model_input = tf.expand_dims(model_input, -1)
        y_true = model_input # regression loss on reconstruction

        if normalize:
            y_true *= 1/(tf.math.reduce_max(y_true) + 1e-7) # lazy normalize
            # model_input *= 1/(tf.math.reduce_max(model_input) + 1e-7)

        with tf.GradientTape() as tape:
            y_pred = self.model(y_true, training=True)
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
        skip_warning = True
        while running:
            time.sleep(0.001)
            if np.sum(self.in_block) > 0.0:
                predicted_block = self.train_step(np.asarray(self.in_block), 
                                                normalize=True)

                self.lock.acquire()
                self.out_block[:] = predicted_block[:]
                self.lock.release()
                self.last_block[:] = predicted_block[:]

                skip_warning = True
            elif skip_warning:
                print("skipping step, input block empty")
                skip_warning = False
                self.out_block[:] = self.last_block * 0.975
                self.last_block[:] = self.out_block[:]
            else:
                self.out_block[:] = self.last_block * 0.975
                self.last_block[:] = self.out_block[:]
        






    # def create_model(self, input_shape, kernel_size=5, latent_dim=4, strides=4):
    #     model = Sequential()
    #     model.add(Conv1D(128, kernel_size, padding='same', strides=strides, activation='tanh', input_shape=input_shape))
    #     # model.add(AveragePooling1D(pool_size=4))
    #     model.add(Conv1D(32, kernel_size, padding='same', strides=strides, activation='tanh'))
    #     # model.add(AveragePooling1D(pool_size=4))
    #     model.add(Conv1D(latent_dim, kernel_size, padding='same', strides=strides, activation='tanh'))
    #     # model.add(AveragePooling1D(pool_size=4))
    #     # model.add(Conv1D(1, kernel_size, padding='same', strides=strides, activation='tanh'))
    #     # model.add(AveragePooling1D(pool_size=4))

    #     # model.add(Flatten())

    #     model.add(Dense(latent_dim, activation=None))

    #     # model.add(Reshape((latent_dim, 1)))

    #     # model.add(Conv1D(1, kernel_size, padding='same', activation='tanh'))
    #     # model.add(Conv1DTranspose(1, kernel_size, padding='same', strides=strides, activation='tanh'))
    #     # model.add(UpSampling1D(size=4))
    #     model.add(Conv1DTranspose(latent_dim, kernel_size, padding='same', strides=strides, activation='tanh'))
    #     # model.add(UpSampling1D(size=4))
    #     model.add(Conv1DTranspose(32, kernel_size, padding='same', strides=strides, activation='tanh'))
    #     # model.add(UpSampling1D(size=4))
    #     model.add(Conv1DTranspose(128, kernel_size, padding='same', strides=strides, activation='tanh'))
    #     # model.add(UpSampling1D(size=4))
    #     model.add(Conv1D(1, kernel_size, padding='same', activation='tanh'))
    #     print(model.summary())
    #     return model


        # def create_model(self, input_shape, kernel_size=5):
        # model = Sequential()
        # model.add(Conv1D(256, kernel_size, strides=4, padding='same', activation='tanh', input_shape=input_shape))
        # # model.add(AveragePooling1D(pool_size=2))
        # model.add(Conv1D(100, kernel_size, strides=4, padding='same', activation='tanh'))
        # # model.add(AveragePooling1D(pool_size=2))
        # model.add(Conv1D(8, kernel_size, strides=4, padding='same', activation='tanh'))
        # # model.add(AveragePooling1D(pool_size=2))
        # model.add(Conv1D(1, kernel_size, padding='same', activation='tanh'))


        # model.add(UpSampling1D(size=4))
        # model.add(Conv1DTranspose(8, kernel_size, padding='same', activation='tanh'))
        # model.add(UpSampling1D(size=4))
        # model.add(Conv1DTranspose(100, kernel_size, padding='same', activation='tanh'))
        # model.add(UpSampling1D(size=4))
        # model.add(Conv1DTranspose(256, kernel_size, padding='same', activation='tanh'))
        # model.add(Conv1DTranspose(1, kernel_size, padding='same', activation='tanh'))
        # print(model.summary())
        # return model