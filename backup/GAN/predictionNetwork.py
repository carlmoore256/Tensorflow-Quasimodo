import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Conv1DTranspose, AveragePooling1D
from tensorflow.keras.layers import Dropout, LayerNormalization, Activation, Softmax, Concatenate, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import multiprocessing
import time
from threading import Thread
import numpy as np

# class PredictionNetwork(multiprocessing.Process):
class PredictionNetwork():
    def __init__(self, block_size, in_block, out_block, seed_size=100, lr=0.001):
        # super(PredictionNetwork, self).__init__()
        self.optimizer = Adam(lr=lr)
        self.MSE = tf.keras.losses.MeanSquaredError()
        self.block_size = block_size
        self.in_block = in_block
        self.out_block = out_block
        self.seed_size = seed_size
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                                    beta_1=0.5,
                                                    beta_2=0.99)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, 
                                                    beta_1=0.5,
                                                    beta_2=0.99)
        self.create_models()

    def build_generator(self, seed_size, dense_size, block_size, k_size=9, stride=4):
        model = Sequential()
        model.add(Dense(dense_size[0]*dense_size[1], activation='relu', input_dim=seed_size))
        model.add(Reshape((dense_size[0],dense_size[1])))
        current_len = model.layers[-1].output_shape[1]
        filters = 256
        while current_len < block_size:
            model.add(Conv1DTranspose(filters, strides=stride, kernel_size=k_size, padding='same'))
            model.add(Activation('tanh'))
            filters = filters//2
            current_len = model.layers[-1].output_shape[1]
        model.add(Conv1DTranspose(1, kernel_size=k_size, padding='same', activation='tanh'))
        return model

    def build_discriminator(self, input_shape, k_size=9):
        model_input = Input(input_shape)
        hidden = Conv1D(32, k_size)(model_input)
        hidden = Activation('tanh')(hidden)
        hidden = MaxPooling1D(pool_size=2)(hidden)
        # hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.1)(hidden)
        hidden = Dense(128)(hidden)
        hidden = Activation('tanh')(hidden)
        hidden = MaxPooling1D(pool_size=2)(hidden)
        # hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.1)(hidden)
        hidden = Dense(256)(hidden)
        hidden = Activation('tanh')(hidden)
        hidden = MaxPooling1D(pool_size=2)(hidden)
        # hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.1)(hidden)
        hidden = Dense(512)(hidden)
        hidden = Activation('tanh')(hidden)
        hidden = AveragePooling1D(pool_size=2)(hidden)
        # hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.1)(hidden)
        hidden = Dense(512)(hidden)
        hidden = Activation('tanh')(hidden)
        hidden = AveragePooling1D(pool_size=2)(hidden)
        # hidden = BatchNormalization()(hidden)
        hidden = Flatten()(hidden)
        output = Dense(1, activation='sigmoid')(hidden)
        return Model(inputs=[model_input], outputs=[output])

    def create_models(self):
        self.generator = self.build_generator(
                                        seed_size=self.seed_size, 
                                        dense_size=[16,16], 
                                        block_size=self.block_size,
                                        k_size=5,
                                        stride=2)
        self.discriminator = self.build_discriminator(
                                        input_shape=[self.block_size, 1], 
                                        k_size=5)

        print(self.generator.summary())
        print(self.discriminator.summary())


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_train_step(self, waveform):
        noise = tf.random.normal([1, self.block_size, 1])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # discriminator train step on noise
            fake_output = self.discriminator(noise, training=True)
            real_output = self.discriminator(waveform, training=True)
            gen_loss = 0
            disc_loss = self.discriminator_loss(real_output, fake_output)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss, noise

    def gan_train_step(self, real_wav):
        seed = tf.random.normal([1, self.seed_size])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_wav = self.generator(seed, training=True)
            fake_output = self.discriminator(gen_wav, training=True)
            real_output = self.discriminator(real_wav, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss, gen_wav


    def train_step(self, model_input, normalize=False):
        model_input = tf.expand_dims(model_input, 0)
        model_input = tf.expand_dims(model_input, -1)
        if normalize:
            model_input *= 1/(tf.math.reduce_max(model_input) + 1e-7)
        gen_loss, disc_loss, gen_wav = self.gan_train_step(model_input)
        print(f'gen_loss: {gen_loss} disc_loss: {disc_loss}')
        return tf.squeeze(gen_wav)

    def start_thread(self):
        runThread = Thread(target=self.run, daemon=True)
        runThread.start()

    def run(self):
        running = True
        while running:
            time.sleep(0.001)
            self.out_block[:] = self.train_step(np.asarray(self.in_block), normalize=True)
        