import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv1DTranspose, AveragePooling1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Dropout, LayerNormalization, Activation, Softmax, Concatenate, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import multiprocessing
import time
from threading import Thread
import numpy as np
from dataWatchdog import DataWatchdog
import glob
import os

class Autoencoder(Model):
    def __init__(self, kernel_size, block_size):
        super(Autoencoder, self).__init__()

        stride=2

        self.encoder = tf.keras.Sequential([
            Input(shape=(block_size,1)), 
            Conv1D(64, kernel_size, strides=stride, padding='same'),
            # AveragePooling1D(pool_size=2),
            Conv1D(64, kernel_size, strides=stride, padding='same'),
            # AveragePooling1D(pool_size=2),
            Conv1D(16, kernel_size, strides=stride, padding='same'),
            # AveragePooling1D(pool_size=2),
            Conv1D(16, kernel_size, strides=stride, padding='same'),
            # AveragePooling1D(pool_size=2),
            Conv1D(4, kernel_size, strides=stride, padding='same'),
            # Conv1D(1, kernel_size, activation='tanh', padding='same',
            # activity_regularizer=tf.keras.regularizers.l1(10e-5))])
            Conv1D(1, kernel_size, activation=None, padding='same')])
            
        
        self.decoder = tf.keras.Sequential([
            Conv1DTranspose(4, kernel_size, strides=stride, activation='tanh', padding='same'),
            # UpSampling1D(size=2),
            Conv1DTranspose(16, kernel_size, strides=stride, activation='tanh', padding='same'),
            # UpSampling1D(size=2),
            Conv1DTranspose(16, kernel_size, strides=stride, activation='tanh', padding='same'),
            # UpSampling1D(size=2),
            Conv1DTranspose(64, kernel_size, strides=stride, activation='tanh', padding='same'),
            # UpSampling1D(size=2),
            Conv1DTranspose(64, kernel_size, strides=stride, activation='tanh', padding='same'),
            Conv1D(1, kernel_size, padding='same', activation='tanh')])

        self.encoded_shape = self.encoder.layers[-1].output_shape
        print(f'ENCODED SHAPE {self.encoded_shape}')

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class PredictionNetwork():
    def __init__(self, block_size, in_block, out_block, 
                    lock, lr=0.001, kernel_size=9, batch_size=8):
        self.save_step = 100
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        self.MSE = tf.keras.losses.MeanAbsoluteError()

        self.batch_size = batch_size

        self.block_size = block_size
        self.in_block = in_block
        self.out_block = out_block
        self.lock = lock
        self.last_block = np.zeros((block_size,))

        self.autoencoder = Autoencoder(kernel_size, block_size)
        self.autoencoder.build((1,block_size, 1))
        print(self.autoencoder.summary())

        init_arr = np.zeros((self.autoencoder.encoded_shape[1], self.autoencoder.encoded_shape[2]))
        self.latent_activation = multiprocessing.Array('d', init_arr)

        self.train_data = []
        # self.dataset_iter = None
        self.dataset = None
        self.has_examples = False

        self.clear_training_cache()
        # initialize watchdog to wait for new blocks to arrive from puredata
        self.watchdog = DataWatchdog("./tmp/train_blocks/", self)

    def clear_training_cache(self):
        files = glob.glob('./tmp/train_blocks/*.wav')
        print(f'removing {len(files)} files in train_blocks cache')
        for f in files:
            os.remove(f)

    def data_generator(self):
        i=0
        while i < len(self.train_data)-1:

            yield self.train_data[i]

    def load_train_data(self, filename):
        raw_audio = tf.io.read_file(filename)
        waveform = tf.audio.decode_wav(raw_audio)[0]
        self.train_data.append(waveform)

        # self.dataset = tf.data.Dataset.from_generator(self.data_generator, output_types=tf.float32)
        # self.dataset = tf.data.Dataset.from_tensors(self.train_data)
        self.dataset = tf.data.Dataset.from_tensor_slices(self.train_data)

        if len(self.train_data) >= self.batch_size:
            # self.dataset = self.dataset.repeat().batch(self.batch_size)
            self.dataset = self.dataset.batch(self.batch_size)

        self.has_examples = True

        # self.dataset = tf.data.Dataset.zip((self.dataset, tf.data.Dataset.from_tensors(waveform)))
        # dataset = tf.data.Dataset.list_files("./tmp/train_blocks/*.wav")

    def predict_step(self, model_input, normalize=True):
        model_input = tf.reshape(model_input, (1, model_input.shape[0], 1))
        model_input = tf.cast(model_input, tf.float32)
        # model_input = tf.expand_dims(model_input, 0)
        # model_input = tf.expand_dims(model_input, -1)
    
        y_true = model_input # regression loss on reconstruction

        # dont calculate gradients/calc loss
        latent, y_pred = self.autoencoder(model_input)
        return tf.squeeze(y_pred), latent


    def train_step(self, normalize=True):

        # for batch in self.dataset.take(self.max_train_examples): # for later
        for batch in self.dataset:

            y_true = batch # regression loss on reconstruction

            if normalize:
                # y_true = (y_true - tf.reduce_min(y_true)) / (tf.reduce_max(y_true) - tf.reduce_min(y_true))
                y_true *= 1/(tf.math.reduce_max(y_true) + 1e-7) # lazy normalize
                # model_input *= 1/(tf.math.reduce_max(model_input) + 1e-7)

            with tf.GradientTape() as tape:
                latent, y_pred = self.autoencoder(batch)
                loss = self.MSE(y_true, y_pred)

            gradients = tape.gradient(loss, self.autoencoder.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables)) # apply gradients to optimizer

            print(f'loss: {loss}')
        return loss
        # return tf.squeeze(y_pred), latent

    def start_thread(self):
        runThread = Thread(target=self.run, daemon=True)
        runThread.start()

    def run(self):
        running = True
        skip_warning = True
        steps = 0
        # try:
        while True:
            time.sleep(0.001)

            # if steps > self.save_step:
            #     self.autoencoder.save('./models/autoencoder.h5')
            #     self.save_step = 0
            # else:
            #     steps +=1

            if np.sum(self.in_block) > 0.0:

                if self.has_examples:
                    loss = self.train_step(normalize=False)

                predicted_block, latent = self.predict_step(np.asarray(self.in_block), normalize=True)

                self.lock.acquire()
                self.out_block[:] = predicted_block[:]
                latent = tf.squeeze(latent)
                self.latent_activation[:] = latent
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
        # except KeyboardInterrupt:
        #     self.watchdog.stop_observer()
        






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

# def create_model(self, input_shape, kernel_size=5):

# encoder = tf.keras.Sequential([
# Input(shape=input_shape), 
# Conv1D(16, kernel_size, strides=2, padding='same'),
# Conv1D(8, kernel_size, strides=2, padding='same'),
# Conv1D(1, kernel_size, activation='tanh', 
# activity_regularizer=tf.keras.regularizers.l1(10e-5))])

# self.encoded_shape = encoder.layers[-1].output_shape

# decoder = tf.keras.Sequential([
# Conv1DTranspose(8, kernel_size, strides=2, activation='tanh', padding='same'),
# Conv1DTranspose(16, kernel_size, strides=2, activation='tanh', padding='same'),
# Conv1D(1, kernel_size, activation='tanh')])


# # input_layer = Input(shape=input_shape)

# # enc = Conv1D(16, kernel_size, strides=2, padding='same')(input_layer)
# # enc = Conv1D(8, kernel_size, strides=2, padding='same')(enc)

# # add sparsity constraint in encoded layer
# # encoded = Conv1D(1, kernel_size, activation='tanh', 
# #     activity_regularizer=tf.keras.regularizers.l1(10e-5))(enc)
# # print(f'ENCODED SHAPE {self.encoded_shape}')


# # dec = Conv1DTranspose(8, kernel_size, strides=2, 
# #     activation='tanh', padding='same')(encoded)
# # dec = Conv1DTranspose(16, kernel_size, strides=2, 
# #     activation='tanh', padding='same')(enc)
# # normal conv layer as shown at 
# # https://www.tensorflow.org/tutorials/generative/autoencoder
# # decoded = Conv1D(1, kernel_size, activation='tanh')(dec)

# # autoencoder = Model(inputs=[input_layer], outputs=[decoded])

# # encoder = Model(input_layer, encoded)

# # encoded_input = Input(shape=self.encoded_shape)
# # decoder_layer = autoencoder.layers[-2:-1]

# # decoder = Model(encoded_input, decoder_layer(encoded_input))
# # decoder = Model(encoded_input, decoder_laye)

# print(encoder.summary())
# print(decoder.summary())

# return encoder, decoder