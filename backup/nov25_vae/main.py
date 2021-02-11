# import tensorflow as tf
from pyo import *
from jackServer import JackServer
from audioStream import AudioServerManager
from predictionNetwork import PredictionNetwork
from VAEnet import VAE
from VAEnetFFT import VAE_FFT
import numpy as np
import multiprocessing
import time


sr = 44100
block_size = 512
channels = 1

init_arr = np.zeros((block_size,))

in_block = multiprocessing.Array('d', init_arr)
out_block = multiprocessing.Array('d', init_arr)
lock = multiprocessing.Lock()


jackManager = JackServer(sr, block_size, 2)
jackManager.system_kill_jack()
jackManager.start_server()

pyoManager = AudioServerManager(sr, block_size, channels)
pyoManager.start_server(in_block, out_block, lock)

VAE_model = VAE(block_size=block_size,
                in_block=in_block,
                out_block=out_block,
                lock=lock,
                lr=0.005)
VAE_model.start_thread()

time.sleep(3600)

pyoManager.kill_server()
jackManager.kill_server()
jackManager.system_kill_jack()
