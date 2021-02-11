import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, Dense, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Conv1DTranspose, Add, AveragePooling1D
from tensorflow.keras.layers import Concatenate, Reshape, Cropping1D, LeakyReLU
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from threading import Thread
import numpy as np
from Synth_MIDI import MidiManager
import time
# import IPython.display as ipd


class SynthAutoencoder():
    def __init__(self, block_size, in_block, out_block, lock, lr, param_controls):
        print(param_controls.shape)
        self.encoder, self.decoder = self.build_model(
            input_shape=(block_size, 1), 
            latent_dim=len(param_controls)
            )
        self.in_block = in_block
        self.out_block = out_block
        self.lock = lock
        self.lr = lr
        self.MIDI_Manager = MidiManager(
            channel = 1,
            param_controls = param_controls,
            device_name = 'minilogue SOUND'
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    def build_model(self, input_shape, latent_dim=4):
        encoder_input = Input(shape=input_shape)
        encoder = Conv1D(320, kernel_size=9, strides=4, activation='relu', padding='same')(encoder_input)
        encoder = Conv1D(160, kernel_size=9, strides=4, activation='relu', padding='same')(encoder)
        encoder = Conv1D(8, kernel_size=9, strides=4, activation='relu', padding='same')(encoder)
        encoder = Conv1D(1, kernel_size=9, activation=None, padding='same')(encoder)
        encoder_output = Dense(latent_dim)(encoder)
        print(encoder_output.shape)

        decoder_input = Input((encoder_output.shape[1], encoder_output.shape[2]))
        decoder = Conv1DTranspose(8, kernel_size=9, strides=4, activation='relu', padding='same')(decoder_input)
        decoder = Conv1DTranspose(160, kernel_size=9, strides=4, activation='relu', padding='same')(decoder)
        decoder = Conv1DTranspose(320, kernel_size=9, strides=4, activation='relu', padding='same')(decoder)
        decoder_output = Conv1D(1, kernel_size=9, strides=1, activation='tanh', padding='same')(decoder)

        return Model(inputs=[encoder_input], outputs=[encoder_output]), Model(inputs=[decoder_input], outputs=[decoder_output])

    def start_thread(self):
        runThread = Thread(target=self.run, daemon=True)
        runThread.start()

    def encode(self, x, training=True):
        encoded = self.encoder(x)
        return encoded

    def decode(self, z, training=True):
        decoded = self.decoder(z)
        return decoded

    def compute_loss(self, y_pred, y_true):
        return tf.keras.losses.mse(y_true, y_pred)

    def train_step(self, x):
        # sends encoded vector to synth <- actually NO
        # encoded 1d vector could be sent to synth for parameters?
        # save latest params into a buffer, have ssh extract those infereces to drive synth
        # model is trying to constantly get synth to resonate with environemnt loop
        with tf.GradientTape() as tape:
            enc = self.encode(x)

            self.MIDI_Manager.set_control_values(enc)

            dec = self.decode(enc)
            # send decoder out to quasimodo, next block recieved is x

            loss = self.compute_loss(dec, x)

        gradients = tape.gradient(loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        return enc, dec

    def run(self):
        running = True
        skip_warning = True
        while running:
            time.sleep(0.001)
            if np.sum(self.in_block) != 0.0:
                input_block = np.asarray(self.in_block)

                enc, dec = self.train_step(input_block)

                self.lock.acquire()
                self.out_block[:] = enc[:]
                self.lock.release()

                skip_warning = True
            elif skip_warning:
                print("skipping step, input block empty")
                skip_warning = False
        

    # encoder, decoder = build_model((1024,1))
    # print(encoder.summary(), decoder.summary())



