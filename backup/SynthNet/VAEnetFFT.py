import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Reshape
from tensorflow.keras.layers import Dropout, LayerNormalization, Activation, Softmax, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
import multiprocessing
import time
from threading import Thread
import numpy as np

class CVAE(tf.keras.Model):
#   Convolutional variational autoencoder
    def __init__(self, latent_dim, block_size, kernel_size=13):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(latent_dim, block_size, kernel_size)
        self.decoder = self.build_decoder(latent_dim, block_size, kernel_size)
        print(self.encoder.summary())
        print(self.decoder.summary())

    def build_encoder(self, latent_dim, block_size, k_size):
        model = Sequential()
        model.add(Input(shape=(block_size, 1)))
        model.add(Conv1D(4, kernel_size=k_size, padding='same', activation='relu'))
        current_len = model.layers[-1].output_shape[1]
        filters = 8
        while current_len > latent_dim*4:
            model.add(Conv1D(filters, strides=1, kernel_size=k_size, padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Activation('relu'))
            filters *= 2
            current_len = model.layers[-1].output_shape[1]

        model.add(Flatten())
        model.add(Dense(latent_dim*2))
        return model

    def build_decoder(self, latent_dim, block_size, k_size):
        model = Sequential()
        model.add(Input(shape=(latent_dim,)))
        model.add(Dense(units=latent_dim * 2, activation='relu'))
        model.add(Reshape(target_shape=(latent_dim, latent_dim)))

        current_len = model.layers[-1].output_shape[1]
        filters = 256
        while current_len < block_size:
            model.add(Conv1DTranspose(filters, strides=1, kernel_size=k_size, padding='same'))
            model.add(UpSampling1D(size=2))
            model.add(Activation('relu'))
            filters = filters//2
            current_len = model.layers[-1].output_shape[1]

        model.add(Conv1DTranspose(1, kernel_size=k_size, padding='same', activation=None))
        return model


    # @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=False)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    

class VAE_FFT():
    def __init__(self, block_size, in_block, out_block, lock, latent_dim=2, lr=0.001):
        self.optimizer = Adam(lr=lr)
        # self.optimizer = tf.keras.optimizers.SGD()
        # don't reduce
        self.MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        # self.model = self.create_model(input_shape=(block_size, 1), kernel_size=15)
        self.in_block = in_block
        self.out_block = out_block
        self.lock = lock
        self.generation_vect = tf.random.normal(shape=[1, latent_dim])
        # divide block size by 2, eliminate fft mirror
        self.model = CVAE(latent_dim, block_size//2)
        self.train_increment = 0
        self.loss_update = 16
        self.total_loss = 0

    def to_complex(self, signal):
        return tf.cast(signal, tf.complex64)

    def audio_to_fft(self, audio, window=True, magnitude=True):
        # audio = tf.expand_dims(audio, 0)
        # print(f'AUDIO SHAPE {audio.shape}')
        if window:
            audio *= tf.signal.hann_window(audio.shape[0])
        # audio = tf.cast(audio, tf.complex64)
        fft = tf.signal.fft(self.to_complex(audio))
        fft = fft[:(len(fft)//2)]
        if magnitude:
            fft = tf.math.abs(fft)
        return fft
    
    def fft_to_audio(self, fft):
        fft = tf.pad(fft, [[0, len(fft)]])
        audio = tf.signal.ifft(self.to_complex(fft))
        audio = tf.cast(audio, tf.float32)
        return audio

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        x = tf.cast(x, tf.float32)
        mean, logvar = self.model.encode(x)
        # print(f'MEAN {mean[0]}')
        # print(f'LOGVAR {logvar[0]}')
        z = self.model.reparameterize(mean, logvar)
        x_logit = self.model.decode(z)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        mse = self.MSE(x, x_logit)

        # print(f'cross ent {cross_ent}')
        # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
        logpx_z = -tf.reduce_sum(mse)
        # print(f'logpx_z {logpx_z}')
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x), x_logit


    def train_step(self, model_input, normalize=False):
        # y_true = tf.random.normal(model_input.shape)
        model_input = tf.expand_dims(model_input, 0)
        model_input = tf.expand_dims(model_input, -1)
        if normalize:
            model_input *= 1.0/(tf.math.reduce_max(model_input) + 1e-7)

        with tf.GradientTape() as tape:
            loss, y_pred = self.compute_loss(model_input)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # apply gradients to optimizer
        
        self.total_loss += loss
        self.train_increment += 1

        # y_pred = self.model.decode(self.generation_vect)

        if self.train_increment == self.loss_update:
            avg_loss = self.total_loss / self.loss_update
            print(f'avg loss: {avg_loss}')
            self.total_loss = 0
            self.train_increment = 0

        return tf.squeeze(y_pred)

    def start_thread(self):
        runThread = Thread(target=self.run, daemon=True)
        runThread.start()

    def run(self):
        running = True
        skip_warning = True
        while running:
            time.sleep(0.001)
            if np.sum(self.in_block) != 0.0:
                input_block = np.asarray(self.in_block)
                input_fft = self.audio_to_fft(input_block, window=True, magnitude=True)

                predicted_fft = self.train_step(input_fft, normalize=False)
                predicted_audio = self.fft_to_audio(predicted_fft)

                self.lock.acquire()
                self.out_block[:] = predicted_audio[:]
                self.lock.release()

                skip_warning = True
            elif skip_warning:
                print("skipping step, input block empty")
                skip_warning = False
        


        # self.combined_logits.append(tf.squeeze(x_logit))
        # self.train_increment+=1
        # if self.train_increment % 30 == 29:
        #     audio = np.zeros(x_logit.shape[1] * len(self.combined_logits))
        #     for i, l in enumerate(self.combined_logits):
        #         audio[i*x_logit.shape[1]:(i*x_logit.shape[1])+x_logit.shape[1]] = l
        #     np.save(f'./saved_logits/x_logits_{self.train_increment}.npy', audio)
        #     print(f'saved npy number {self.train_increment}')

        # self.encoder = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=(block_size,1)),
        #     tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size, strides=2, activation='tanh'),
        #     tf.keras.layers.Conv1D(filters=64, kernel_size=kernel_size, strides=2, activation='tanh'),
        #     tf.keras.layers.Flatten(),
        #     # No activation
        #     tf.keras.layers.Dense(latent_dim + latent_dim),])


        # self.encoder = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=input_shape),
        #     tf.keras.layers.Conv1D(filters=32, kernel_size=kernel_size, strides=2, activation='tanh'),
        #     tf.keras.layers.Conv1D(filters=64, kernel_size=kernel_size, strides=2, activation='tanh'),
        #     tf.keras.layers.Flatten(),
        #     # No activation
        #     tf.keras.layers.Dense(latent_dim + latent_dim),])

        # print(self.encoder.summary())

        # self.decoder = self.build_decoder(latent_dim, block)
        # print(self.decoder.summary())
        # self.decoder = tf.keras.Sequential([
            # tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            # tf.keras.layers.Dense(units=latent_dim * 2, activation=tf.nn.relu),
            # tf.keras.layers.Reshape(target_shape=(latent_dim, latent_dim)),
            # tf.keras.layers.Conv1DTranspose(
            #     filters=64, kernel_size=kernel_size, strides=2, padding='same',
            #     activation='tanh'),
            # tf.keras.layers.Conv1DTranspose(
            #     filters=32, kernel_size=kernel_size, strides=2, padding='same',
            #     activation='tanh'),
            # tf.keras.layers.Conv1DTranspose(
            #     filters=16, kernel_size=kernel_size, strides=2, padding='same',
            #     activation='tanh'),
            # # No activation
            # tf.keras.layers.Conv1DTranspose(
            #     filters=1, kernel_size=kernel_size, strides=1, padding='same'),])